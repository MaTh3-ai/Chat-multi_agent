"""Orchestrateur : le Chef de projet et la boucle d'états déterministe.

Le Chef de projet (tier expert) est le SEUL interlocuteur de l'utilisateur.
La boucle est un automate Python déterministe :

    INTERVIEW  --(user valide le besoin)-->  PLANIF
    PLANIF     --(user valide le plan)   -->  EXEC
    EXEC       --(toutes tâches faites)  -->  SYNTHESE
    SYNTHESE   --(retours user)          -->  PLANIF / fin

Les décisions "molles" (questions, plan, synthèse) sont déléguées au LLM.
Les transitions "dures" (qui avance, quand, jusqu'où) sont en Python pur :
le modèle ne pilote jamais la boucle, ce qui évite dérives et coûts fous.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.models import (
    CHIEF_TIER,
    resolve_tier_to_model_alias,
    get_tier,
    sanitize_temperature,
)
from llm_engine import create_chat_model

from core.multi_agent.ledger import (
    TaskLedger,
    STATE_INTERVIEW,
    STATE_PLANIF,
    STATE_EXEC,
    STATE_SYNTHESE,
)
from core.multi_agent.roles import (
    chief_system_prompt,
    chief_planning_instruction,
)
from core.multi_agent.planning import build_ledger, PlanReport, PlanError
from core.multi_agent.workers import (
    run_tasks_parallel,
    apply_results,
    _extract_text,
    WorkerResult,
    MAX_PARALLEL_WORKERS,
)


# Température du Chef : un peu de latitude pour l'interview et la synthèse,
# tout en restant cadré.
CHIEF_TEMPERATURE = 0.3
CHIEF_MAX_TOKENS = 4096

# Garde-fou : nombre maximal de vagues d'exécution parallèle. Empêche une
# boucle EXEC infinie si des tâches restent bloquées (ex. toutes "à revoir").
MAX_EXEC_ROUNDS = 12


@dataclass
class Event:
    """Entrée du journal d'exécution, destinée au VISUEL de l'UI."""

    kind: str            # "info" | "plan" | "worker" | "error" | "state"
    message: str
    ts: float = field(default_factory=time.time)


class ChefDeProjet:
    """Orchestrateur principal du système multi-agent.

    Détient le ledger (état partagé) et le journal d'événements. Expose des
    méthodes fines (interview / plan / exécution / synthèse) pour permettre
    à l'UI d'insérer une validation humaine entre chaque étape.
    """

    def __init__(self) -> None:
        self.ledger = TaskLedger(state=STATE_INTERVIEW)
        self.events: list[Event] = []
        self.history: list[tuple[str, str]] = []  # dialogue chef <-> user
        self._chief = None  # modèle expert, instancié à la demande

    # --- Journal -----------------------------------------------------------
    def log(self, kind: str, message: str) -> None:
        self.events.append(Event(kind=kind, message=message))

    # --- Accès au modèle Chef (tier expert) --------------------------------
    def _chief_model(self):
        if self._chief is None:
            alias = resolve_tier_to_model_alias(CHIEF_TIER)
            safe_temperature = sanitize_temperature(alias, CHIEF_TEMPERATURE)
            self._chief = create_chat_model(
                model_alias=alias,
                temperature=safe_temperature,
                max_tokens=CHIEF_MAX_TOKENS,
            )
            self.log("info", f"Chef de projet prêt (modèle : {alias}).")
        return self._chief

    def _ask_chief(self, extra_messages: list[tuple[str, str]]) -> str:
        """Interroge le Chef en lui donnant le système + l'historique."""
        messages: list[tuple[str, str]] = [("system", chief_system_prompt())]
        messages.extend(self.history)
        messages.extend(extra_messages)
        response = self._chief_model().invoke(messages)
        return _extract_text(response)

    # --- Étape INTERVIEW ---------------------------------------------------
    def interview(self, user_message: str) -> str:
        """Un tour de dialogue de clarification avec l'utilisateur."""
        self.ledger.state = STATE_INTERVIEW
        self.history.append(("user", user_message))
        reply = self._ask_chief([])
        self.history.append(("assistant", reply))
        return reply

    # --- Étape PLANIF ------------------------------------------------------
    def plan(self) -> PlanReport:
        """Demande le plan JSON au Chef puis construit le ledger.

        Ne lance PAS l'exécution : l'UI doit d'abord faire valider le plan
        par l'utilisateur (human-in-the-loop).
        """
        self.ledger.state = STATE_PLANIF
        raw = self._ask_chief([("user", chief_planning_instruction())])
        try:
            ledger, report = build_ledger(raw)
        except PlanError as exc:
            self.log("error", f"Plan rejeté : {exc}")
            raise

        # On conserve l'objectif éventuellement affiné pendant l'interview.
        if not ledger.objective:
            ledger.objective = self.ledger.objective
        self.ledger = ledger
        self.log(
            "plan",
            f"Plan validé : {report.task_count} tâche(s), "
            f"répartition {report.tiers_used}.",
        )
        for corr in report.corrections:
            self.log("info", f"Ajustement auto : {corr}")
        return report

    # --- Étape EXEC --------------------------------------------------------
    def execute(self, max_parallel: int = MAX_PARALLEL_WORKERS) -> None:
        """Exécute toutes les tâches par vagues parallèles jusqu'à épuisement.

        Chaque vague = un lot ready_tasks() (tâches mutuellement
        indépendantes) exécuté en parallèle, puis appliqué au ledger.
        Un garde-fou (MAX_EXEC_ROUNDS + détection de non-progression)
        empêche toute boucle infinie.
        """
        self.ledger.state = STATE_EXEC
        rounds = 0
        while not self.ledger.all_done():
            ready = self.ledger.ready_tasks()
            if not ready:
                # Plus rien de prêt alors que tout n'est pas fait :
                # tâches en "à revoir" ou bloquées. On arrête proprement.
                if self.ledger.pending_count() > 0:
                    self.log(
                        "error",
                        "Exécution stoppée : des tâches restent bloquées "
                        "(à revoir ou dépendances non satisfaites).",
                    )
                break

            rounds += 1
            if rounds > MAX_EXEC_ROUNDS:
                self.log("error", "Nombre maximal de vagues atteint.")
                break

            titles = ", ".join(t.title for t in ready)
            self.log("state", f"Vague {rounds} : {len(ready)} tâche(s) — {titles}")

            results = run_tasks_parallel(self.ledger, ready, max_parallel)
            apply_results(self.ledger, results)
            self._log_results(results)

    def start_execution(self) -> None:
        """Passe en phase EXEC sans exécuter aucune vague.

        Permet à l'UI d'entrer dans l'état EXEC puis de dérouler les vagues
        une par une (via execute_one_wave) en rendant la main entre chaque,
        pour offrir une visibilité en temps réel sur la progression.
        """
        self.ledger.state = STATE_EXEC
        self._exec_rounds = 0

    def execute_one_wave(self, max_parallel: int = MAX_PARALLEL_WORKERS) -> bool:
        """Exécute UNE seule vague de tâches prêtes et rend la main.

        Retourne True s'il reste potentiellement du travail (l'UI doit
        rappeler cette méthode après un rerun), False si l'exécution est
        terminée ou définitivement bloquée (l'UI peut passer à la synthèse).

        C'est la version « pas à pas » de execute() : même logique de
        vagues parallèles et mêmes garde-fous, mais une vague par appel.
        """
        if self.ledger.state != STATE_EXEC:
            self.ledger.state = STATE_EXEC
        if not hasattr(self, "_exec_rounds"):
            self._exec_rounds = 0

        if self.ledger.all_done():
            return False

        ready = self.ledger.ready_tasks()
        if not ready:
            if self.ledger.pending_count() > 0:
                self.log(
                    "error",
                    "Exécution stoppée : des tâches restent bloquées "
                    "(à revoir ou dépendances non satisfaites).",
                )
            return False

        self._exec_rounds += 1
        if self._exec_rounds > MAX_EXEC_ROUNDS:
            self.log("error", "Nombre maximal de vagues atteint.")
            return False

        titles = ", ".join(t.title for t in ready)
        self.log(
            "state",
            f"Vague {self._exec_rounds} : {len(ready)} tâche(s) — {titles}",
        )

        results = run_tasks_parallel(self.ledger, ready, max_parallel)
        apply_results(self.ledger, results)
        self._log_results(results)

        return not self.ledger.all_done()

    def _log_results(self, results: list[WorkerResult]) -> None:
        for res in results:
            task = self.ledger.get_task(res.task_id)
            title = task.title if task else res.task_id
            if res.ok:
                self.log(
                    "worker",
                    f"✅ {title} — livré par {res.tier} ({res.model_alias}).",
                )
            else:
                self.log("error", f"❌ {title} — {res.error}")

    # --- Étape SYNTHESE ----------------------------------------------------
    def synthesize(self) -> str:
        """Agrège les livrables en une réponse finale pour l'utilisateur."""
        self.ledger.state = STATE_SYNTHESE
        deliverables: list[str] = []
        for task in self.ledger.tasks:
            if task.deliverable:
                deliverables.append(f"### {task.title}\n{task.deliverable}")
        joined = "\n\n".join(deliverables) if deliverables else "(aucun livrable)"

        instruction = (
            "Toutes les tâches ont été exécutées. Voici les livrables "
            "produits par les workers :\n\n"
            f"{joined}\n\n"
            "Rédige MAINTENANT la réponse finale pour l'utilisateur : "
            "cohérente, vérifiée et directement exploitable. Corrige les "
            "incohérences entre livrables si nécessaire."
        )
        final = self._ask_chief([("user", instruction)])
        self.ledger.final_answer = final
        self.history.append(("assistant", final))
        self.log("state", "Synthèse produite.")
        return final

    # --- Retours utilisateur (nouvelle itération) --------------------------
    def apply_feedback(self, feedback: str) -> None:
        """Enregistre un retour utilisateur et rouvre une phase de PLANIF.

        L'objectif et les décisions sont conservés ; l'utilisateur pourra
        relancer plan() pour un plan révisé tenant compte du retour.
        """
        self.history.append(("user", f"Retour sur la réponse : {feedback}"))
        self.ledger.log_decision(f"Retour utilisateur : {feedback}")
        self.ledger.state = STATE_PLANIF
        self.log("info", "Retour utilisateur pris en compte (re-planification).")


def new_chief() -> ChefDeProjet:
    """Fabrique un orchestrateur neuf (pratique pour l'UI / les tests)."""
    return ChefDeProjet()
