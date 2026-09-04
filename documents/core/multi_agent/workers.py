"""Exécution des workers du système multi-agent.

Un worker :
1. reçoit UNE tâche du ledger (choisie par l'orchestrateur) ;
2. traduit son tier en modèle réel (menu fermé) ;
3. produit un livrable via un appel LLM ;
4. écrit ce livrable DANS le ledger (communication en étoile).

Les workers ne se parlent jamais entre eux : ils ne connaissent que leur
tâche et le contexte que l'orchestrateur leur transmet.

Cette couche est conçue pour le PARALLÉLISME : un lot de tâches
mutuellement indépendantes (ledger.ready_tasks()) peut être exécuté
simultanément via run_tasks_parallel().
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from core.models import resolve_tier_to_model_alias, get_tier
from llm_engine import create_chat_model

from core.multi_agent.ledger import Task, TaskLedger, STATUS_DOING
from core.multi_agent.roles import worker_system_prompt, worker_task_prompt


# Température des workers : basse pour des livrables stables et reproductibles.
WORKER_TEMPERATURE = 0.2

# Nombre maximal de workers exécutés simultanément (borne de sécurité :
# throttling Bedrock + coût). Ajustable côté UI plus tard.
MAX_PARALLEL_WORKERS = 4


@dataclass
class WorkerResult:
    """Résultat brut d'un worker pour une tâche donnée."""

    task_id: str
    tier: str
    model_alias: str
    deliverable: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _extract_text(response: object) -> str:
    """Extrait le texte d'une réponse ChatBedrockConverse.

    .content peut être une chaîne ou une liste de blocs (multimodal).
    On concatène uniquement les blocs texte.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts).strip()
    return str(content).strip()


def run_single_task(task: Task, context: str = "") -> WorkerResult:
    """Exécute UNE tâche avec le modèle correspondant à son tier.

    Cette fonction est pure vis-à-vis du ledger : elle ne le modifie pas,
    elle renvoie un WorkerResult. C'est l'orchestrateur qui applique le
    résultat au ledger (source de vérité unique). Cela la rend sûre à
    lancer dans un thread.
    """
    tier_conf = get_tier(task.assigned_tier)
    try:
        model_alias = resolve_tier_to_model_alias(task.assigned_tier)
        safe_temperature = sanitize_temperature(model_alias, WORKER_TEMPERATURE)
        model = create_chat_model(
            model_alias=model_alias,
            temperature=safe_temperature,
        )
        messages = [
            ("system", worker_system_prompt(tier_conf["label"])),
            ("user", worker_task_prompt(task.title, task.description, context)),
        ]
        response = model.invoke(messages)
        deliverable = _extract_text(response)
        return WorkerResult(
            task_id=task.id,
            tier=task.assigned_tier,
            model_alias=model_alias,
            deliverable=deliverable,
        )
    except Exception as exc:  # noqa: BLE001 - on remonte l'erreur au ledger
        return WorkerResult(
            task_id=task.id,
            tier=task.assigned_tier,
            model_alias=tier_conf.get("model_alias", "?"),
            deliverable="",
            error=f"{type(exc).__name__} : {exc}",
        )


def build_context(ledger: TaskLedger, task: Task) -> str:
    """Construit le contexte transmis au worker.

    On ne transmet que les livrables des tâches dont dépend la tâche
    courante : chaque worker reçoit le strict nécessaire (coût et clarté),
    jamais tout le ledger.
    """
    parts: list[str] = []
    for dep_id in task.dependencies:
        dep = ledger.get_task(dep_id)
        if dep is not None and dep.deliverable:
            parts.append(f"### {dep.title}\n{dep.deliverable}")
    return "\n\n".join(parts)


def run_tasks_parallel(
    ledger: TaskLedger,
    tasks: list[Task],
    max_workers: int = MAX_PARALLEL_WORKERS,
) -> list[WorkerResult]:
    """Exécute EN PARALLÈLE un lot de tâches indépendantes.

    Le lot doit provenir de ledger.ready_tasks() : ses tâches ne dépendent
    pas les unes des autres, l'ordre d'exécution n'a donc aucune importance.

    Le contexte de chaque tâche est figé AVANT le lancement des threads
    (les dépendances sont déjà terminées), ce qui évite tout accès concurrent
    en lecture/écriture au ledger pendant l'exécution parallèle.
    """
    if not tasks:
        return []

    # Marque les tâches "en cours" et fige leur contexte (thread-safe).
    prepared: list[tuple[Task, str]] = []
    for task in tasks:
        ledger.set_status(task.id, STATUS_DOING)
        prepared.append((task, build_context(ledger, task)))

    results: list[WorkerResult] = []
    workers = max(1, min(max_workers, len(prepared)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_single_task, task, ctx): task
            for task, ctx in prepared
        }
        for future in as_completed(futures):
            results.append(future.result())

    return results


def apply_results(ledger: TaskLedger, results: list[WorkerResult]) -> None:
    """Applique au ledger les résultats des workers (mono-thread).

    Séparé de l'exécution pour garder un point d'écriture unique et
    déterministe sur la source de vérité.
    """
    for res in results:
        if res.ok:
            ledger.set_deliverable(res.task_id, res.deliverable)
        else:
            task = ledger.get_task(res.task_id)
            if task is not None:
                task.notes = (task.notes + f"\n[ERREUR] {res.error}").strip()
                # Reste à faire pour permettre une relance ciblée.
                from core.multi_agent.ledger import STATUS_REVIEW
                task.status = STATUS_REVIEW
