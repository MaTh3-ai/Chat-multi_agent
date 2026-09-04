"""Étape PLANIF : transformation du plan du Chef en TaskLedger.

Le Chef de projet (tier expert) produit un plan au format JSON strict
(voir roles.PLAN_JSON_SCHEMA). Ce module :

1. extrait le JSON même s'il est entouré de texte ou de balises ```json ;
2. le valide (structure, tiers assignables, indices de dépendances) ;
3. génère les identifiants de tâches côté Python (jamais côté modèle) ;
4. construit un TaskLedger prêt pour la phase EXEC.

Aucun appel LLM ici : ce module est pur et testable seul. L'orchestrateur
lui fournit le texte brut renvoyé par le modèle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.multi_agent.ledger import Task, TaskLedger, STATE_EXEC
from core.multi_agent.roles import ASSIGNABLE_TIERS


class PlanError(ValueError):
    """Plan invalide ou impossible à interpréter."""


@dataclass
class PlanReport:
    """Compte rendu de la construction du plan (pour l'UI / le journal)."""

    task_count: int
    tiers_used: dict[str, int]        # tier -> nombre de tâches
    corrections: list[str]            # ajustements automatiques appliqués


# --- Extraction du JSON -----------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_block(raw: str) -> str:
    """Isole le bloc JSON dans une réponse potentiellement bruitée.

    Tolère : balises ```json ... ```, texte avant/après, ou JSON nu.
    Stratégie : on tente la balise de code, sinon on prend du premier '{'
    à la dernière '}'.
    """
    if not raw or not raw.strip():
        raise PlanError("Réponse de planification vide.")

    fenced = _FENCE_RE.search(raw)
    if fenced:
        return fenced.group(1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise PlanError("Aucun objet JSON détecté dans le plan.")
    return raw[start : end + 1].strip()


def parse_plan(raw: str) -> dict:
    """Parse le texte brut du Chef en dict Python validé structurellement."""

    block = extract_json_block(raw)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        raise PlanError(f"JSON du plan invalide : {exc}") from exc

    if not isinstance(data, dict):
        raise PlanError("Le plan doit être un objet JSON.")
    if "tasks" not in data or not isinstance(data["tasks"], list):
        raise PlanError("Le plan doit contenir une liste 'tasks'.")
    if not data["tasks"]:
        raise PlanError("Le plan ne contient aucune tâche.")
    return data


# --- Construction du ledger -------------------------------------------------
def _make_task_id(index: int) -> str:
    """Identifiant stable et lisible généré côté Python (jamais par le LLM)."""
    return f"T{index + 1:02d}"


def _normalize_tier(raw_tier: object, corrections: list[str], where: str) -> str:
    """Ramène un tier au menu fermé assignable, avec repli sûr sur 'fast'."""
    tier = str(raw_tier).strip().lower() if raw_tier is not None else ""
    if tier in ASSIGNABLE_TIERS:
        return tier
    corrections.append(
        f"{where} : tier '{raw_tier}' non assignable → repli sur 'fast'."
    )
    return "fast"


def _resolve_dependencies(
    raw_deps: object,
    self_index: int,
    task_count: int,
    corrections: list[str],
    where: str,
) -> list[int]:
    """Valide les indices de dépendances (entiers, in-range, non circulaires)."""
    if raw_deps is None:
        return []
    if not isinstance(raw_deps, list):
        corrections.append(f"{where} : 'depends_on' ignoré (format inattendu).")
        return []

    valid: list[int] = []
    for dep in raw_deps:
        if not isinstance(dep, int):
            corrections.append(f"{where} : dépendance '{dep}' ignorée (non entière).")
            continue
        if dep == self_index:
            corrections.append(f"{where} : auto-dépendance ignorée.")
            continue
        if dep < 0 or dep >= task_count:
            corrections.append(f"{where} : dépendance {dep} hors limites, ignorée.")
            continue
        if dep not in valid:
            valid.append(dep)
    return valid


def build_ledger(raw_plan: str) -> tuple[TaskLedger, PlanReport]:
    """Construit un TaskLedger validé à partir du texte brut du Chef.

    - Génère les id côté Python (T01, T02, ...).
    - Remappe 'depends_on' (indices dans la liste) vers les id générés.
    - Force chaque tier dans le menu assignable (repli 'fast' sinon).
    - Renvoie un PlanReport listant les corrections automatiques.
    """
    data = parse_plan(raw_plan)
    raw_tasks = data["tasks"]
    task_count = len(raw_tasks)
    corrections: list[str] = []

    # Table indices -> id générés (avant de créer les Task pour les deps).
    index_to_id = {i: _make_task_id(i) for i in range(task_count)}

    ledger = TaskLedger(
        objective=str(data.get("objective", "")).strip(),
        state=STATE_EXEC,
    )
    tiers_used: dict[str, int] = {}

    for i, raw_task in enumerate(raw_tasks):
        where = f"Tâche {i + 1}"
        if not isinstance(raw_task, dict):
            raise PlanError(f"{where} : format de tâche invalide.")

        title = str(raw_task.get("title", "")).strip() or f"Tâche {i + 1}"
        description = str(raw_task.get("description", "")).strip()
        if not description:
            raise PlanError(f"{where} : description manquante.")

        tier = _normalize_tier(raw_task.get("assigned_tier"), corrections, where)
        dep_indices = _resolve_dependencies(
            raw_task.get("depends_on"), i, task_count, corrections, where
        )
        dependencies = [index_to_id[d] for d in dep_indices]

        ledger.add_task(
            Task(
                id=index_to_id[i],
                title=title,
                description=description,
                assigned_tier=tier,
                dependencies=dependencies,
            )
        )
        tiers_used[tier] = tiers_used.get(tier, 0) + 1

    _check_no_cycle(ledger)

    report = PlanReport(
        task_count=task_count,
        tiers_used=tiers_used,
        corrections=corrections,
    )
    return ledger, report


def _check_no_cycle(ledger: TaskLedger) -> None:
    """Détecte un cycle de dépendances (bloquerait EXEC indéfiniment)."""
    id_to_deps = {t.id: list(t.dependencies) for t in ledger.tasks}
    visited: dict[str, int] = {}  # 0=en cours, 1=fini

    def dfs(node: str) -> None:
        state = visited.get(node)
        if state == 1:
            return
        if state == 0:
            raise PlanError(f"Cycle de dépendances détecté autour de '{node}'.")
        visited[node] = 0
        for dep in id_to_deps.get(node, []):
            dfs(dep)
        visited[node] = 1

    for task_id in id_to_deps:
        dfs(task_id)
