"""Task Ledger : état partagé et source de vérité du système multi-agent.

Aucune dépendance LLM ici : ce module est pur Python et testable seul.
La communication se fait EN ÉTOILE : chaque worker lit sa tâche et écrit
son livrable dans le ledger ; les workers ne se parlent jamais entre eux.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# --- États de la machine (boucle orchestrateur déterministe) ---------------
STATE_INTERVIEW = "INTERVIEW"
STATE_PLANIF = "PLANIF"
STATE_EXEC = "EXEC"
STATE_SYNTHESE = "SYNTHESE"

# --- Statuts d'une tâche ----------------------------------------------------
STATUS_TODO = "à faire"
STATUS_DOING = "en cours"
STATUS_DONE = "terminé"
STATUS_REVIEW = "à revoir"


@dataclass
class Task:
    """Une unité de travail assignée à un worker."""

    id: str
    title: str
    description: str
    assigned_tier: str                     # "nano" | "fast" | "smart"
    status: str = STATUS_TODO
    dependencies: list[str] = field(default_factory=list)
    deliverable: str | None = None         # livrable produit DANS le ledger
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskLedger:
    """Conteneur unique de l'état partagé du chantier."""

    objective: str = ""
    state: str = STATE_INTERVIEW
    tasks: list[Task] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    final_answer: str | None = None

    # --- Accès / requêtes --------------------------------------------------
    def get_task(self, task_id: str) -> Task | None:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def is_done(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        return task is not None and task.status == STATUS_DONE

    def ready_tasks(self) -> list[Task]:
        """Tâches exécutables maintenant.

        Une tâche est prête si elle est à faire et que TOUTES ses
        dépendances sont terminées. Le lot retourné ne contient que des
        tâches mutuellement indépendantes : il peut donc être exécuté en
        PARALLÈLE sans risque (point d'entrée du parallélisme).
        """
        ready: list[Task] = []
        for task in self.tasks:
            if task.status != STATUS_TODO:
                continue
            if all(self.is_done(dep) for dep in task.dependencies):
                ready.append(task)
        return ready

    def all_done(self) -> bool:
        """Vrai si toutes les tâches sont terminées."""
        return bool(self.tasks) and all(
            t.status == STATUS_DONE for t in self.tasks
        )

    def pending_count(self) -> int:
        return sum(
            1 for t in self.tasks
            if t.status in (STATUS_TODO, STATUS_DOING)
        )

    # --- Mutations ---------------------------------------------------------
    def add_task(self, task: Task) -> None:
        if self.get_task(task.id) is not None:
            raise ValueError(f"Tâche déjà présente : {task.id}")
        self.tasks.append(task)

    def set_status(self, task_id: str, status: str) -> None:
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Tâche inconnue : {task_id}")
        task.status = status

    def set_deliverable(self, task_id: str, content: str) -> None:
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Tâche inconnue : {task_id}")
        task.deliverable = content
        task.status = STATUS_DONE

    def log_decision(self, decision: str) -> None:
        self.decisions.append(decision)

    # --- Visuel : structure de l'équipe -----------------------------------
    def team_structure(self) -> dict[str, list[Task]]:
        """Regroupe les tâches par tier assigné.

        Alimente le VISUEL final : montre l'équipe formée par le Chef,
        c'est-à-dire quels tiers (donc quels modèles) travaillent et sur
        quelles tâches. Le mapping tier -> modèle est résolu côté UI via
        core.models.resolve_tier_to_model_alias().
        """
        grouped: dict[str, list[Task]] = {}
        for task in self.tasks:
            grouped.setdefault(task.assigned_tier, []).append(task)
        return grouped

    # --- Sérialisation (persistance session / debug) ----------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "state": self.state,
            "tasks": [t.to_dict() for t in self.tasks],
            "decisions": list(self.decisions),
            "final_answer": self.final_answer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskLedger":
        ledger = cls(
            objective=data.get("objective", ""),
            state=data.get("state", STATE_INTERVIEW),
            decisions=list(data.get("decisions", [])),
            final_answer=data.get("final_answer"),
        )
        for raw in data.get("tasks", []):
            ledger.tasks.append(Task(**raw))
        return ledger
