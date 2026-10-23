"""Execution robuste et parallele des workers du systeme multi-agent."""
from __future__ import annotations

import random
import json
import threading
import time
import hashlib
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from langchain_core.messages import ToolMessage, HumanMessage, RemoveMessage
from core.agent_runtime import CURRENT_EXECUTION, ExecutionSession, RunBudget, UsageRecorder, path_lock
from tools.custom_tools import PROJECT_ROOT, WORKSPACE_ROOT, resolve_workspace_path

from core.models import get_tier, resolve_tier_to_model_alias, sanitize_temperature
from core.multi_agent.ledger import STATUS_CANCELLED, Task, TaskLedger, TaskMetrics
from core.multi_agent.coordination import compile_context, workspace_revision, task_paths
from core.multi_agent.roles import worker_system_prompt, worker_task_prompt
from llm_engine import create_chat_model, get_workspace_editor_agent, get_workspace_reader_agent

# Rapport d'avancement temps reel. Signature : (kind, message, task_id).
# Volontairement un simple callable plutot qu'un type importe de background.py :
# workers.py ne doit dependre ni de l'UI, ni du moteur de taches de fond.
ProgressCallback = Callable[[str, str, "str | None"], None]


def _emit(progress: ProgressCallback | None, kind: str, message: str, task_id: str | None = None) -> None:
    """Publie un evenement d'avancement sans jamais faire echouer l'execution."""
    if progress is None:
        return
    try:
        progress(kind, message, task_id)
    except Exception:
        # Un observateur defaillant ne doit pas interrompre un worker.
        pass


# Limite de recursion LangGraph pour les agents workers (lecture / edition).
# Auparavant implicite (defaut LangGraph = 25), ce qui provoquait des
# GraphRecursionError sur les taches necessitant plusieurs appels d'outils.
# Alignee sur un plafond eleve, coherent avec la boucle de chat.
AGENT_RECURSION_LIMIT = 60

# Nombre maximal de relances lorsqu'un modele s'arrete sur stopReason
# "max_tokens". Chaque relance demande la suite EXACTE de la sortie, ce qui
# evite les livrables coupes en plein milieu (et les workers qui interpretent
# a tort cette coupure comme une erreur).
MAX_CONTINUATIONS = 3

# Objectif de sobriete, jamais utilise comme condition d'arret. Un worker peut
# et doit le depasser si cela est necessaire pour terminer correctement.
TARGET_OUTPUT_TOKENS = 1200

CONTINUATION_INSTRUCTION = (
    "Ta reponse precedente a ete coupee par la limite de tokens. "
    "Reprends EXACTEMENT au caractere suivant celui ou tu t'es arrete. "
    "Ne repete rien, n'ajoute ni preambule, ni excuse, ni resume, et ne "
    "signale pas la coupure : continue simplement le texte."
)

# Valeurs de stopReason signalant une sortie tronquee, selon les providers
# exposes par Bedrock Converse.
_TRUNCATION_STOP_REASONS = {"max_tokens", "max_token", "length", "max_output_tokens"}


# Indices textuels indiquant qu'une tache exige une lecture reelle du workspace
# (et non une simple exploitation du contexte pre-lu). Quand l'un d'eux apparait
# dans le titre ou la description, le worker recoit un agent de LECTURE SEULE
# capable d'ouvrir reellement les fichiers via les outils de consultation.
_READ_INTENT_TOKENS = (
    "lire",
    "lis ",
    "lecture",
    "ouvrir le fichier",
    "ouvre le fichier",
    "consulter le fichier",
    "consulte le fichier",
    "contenu du fichier",
    "contenu des fichiers",
    "inspecter le fichier",
    "read_workspace_file",
    "lire_fichier",
)

# Indices textuels indiquant qu'une tache exige de MODIFIER le workspace (et pas
# seulement de le lire). Quand l'un d'eux apparait dans le titre ou la
# description, le worker recoit un agent capable de creer des PROPOSITIONS
# (proposer_*), comme le chatbot. Aucune ecriture directe : la proposition
# reste soumise a approbation dans Streamlit.
_WRITE_INTENT_TOKENS = (
    "implement", "update", "fix ", "modify", "create ", "edit ", "write ", "delete ", "remove ",
    "modifie",
    "modifier",
    "corrige",
    "corriger",
    "ajoute",
    "ajouter",
    "supprime",
    "supprimer",
    "remplace",
    "remplacer",
    "insere",
    "inserer",
    "cree le fichier",
    "creer le fichier",
    "ecris",
    "ecrire",
    "refactor",
    "renomme",
    "patch",
    "proposer_",
    "proposition de modification",
)


WORKER_TEMPERATURE = 0.2
MAX_PARALLEL_WORKERS = 4
MAX_PARALLEL_PER_MODEL = 2
MAX_ATTEMPTS = 3
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 8.0
JITTER_SECONDS = 0.35

_RETRYABLE_ERROR_NAMES = {"ThrottlingException", "ModelTimeoutException", "ServiceUnavailableException", "InternalServerException", "ReadTimeoutError", "ConnectTimeoutError", "ConnectionError", "TimeoutError"}
_NON_RETRYABLE_ERROR_NAMES = {"AccessDeniedException", "ValidationException", "ResourceNotFoundException", "ValueError", "TypeError", "KeyError", "NameError", "SyntaxError"}
_SEMAPHORE_LOCK = threading.Lock()
_MODEL_SEMAPHORES: dict[str, threading.BoundedSemaphore] = {}
_TASK_PATH_LOCKS: dict[str, Any] = {}


@dataclass(frozen=True)
class WorkerExecutionConfig:
    max_parallel_workers: int = MAX_PARALLEL_WORKERS
    max_parallel_per_model: int = MAX_PARALLEL_PER_MODEL
    max_attempts: int = MAX_ATTEMPTS
    base_backoff_seconds: float = BASE_BACKOFF_SECONDS
    max_backoff_seconds: float = MAX_BACKOFF_SECONDS
    jitter_seconds: float = JITTER_SECONDS
    budget: RunBudget | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_parallel_workers < 1 or self.max_parallel_per_model < 1:
            raise ValueError("Les limites de concurrence doivent etre positives.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts doit etre superieur ou egal a 1.")
        if min(self.base_backoff_seconds, self.max_backoff_seconds, self.jitter_seconds) < 0:
            raise ValueError("Les delais de retry ne peuvent pas etre negatifs.")


@dataclass
class WorkerResult:
    task_id: str
    tier: str
    model_alias: str = "?"
    deliverable: str = ""
    error: str | None = None
    error_type: str | None = None
    retryable: bool = False
    attempt_count: int = 0
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    latency_ms: float | None = None
    truncated: bool = False
    continuations: int = 0
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    input_revision: str = ""
    output_revision: str = ""
    tool_evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.deliverable.strip())


def extract_text(response: object, strip: bool = True) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip() if strip else content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        text = "\n".join(parts)
        return text.strip() if strip else text
    return str(content).strip()


# Compatibilite temporaire avec orchestrator.py actuel.
_extract_text = extract_text


def _usage_from_response(response: object) -> TaskMetrics:
    usage = getattr(response, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        usage = {}
    return TaskMetrics(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"), total_tokens=usage.get("total_tokens"), model_calls=1)


def _usage_from_messages(messages: Any) -> TaskMetrics:
    """Additionne l'usage de tous les messages modele d'un etat d'agent."""
    metrics = TaskMetrics()
    calls = 0
    for message in messages or []:
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            continue
        calls += 1
        metrics.input_tokens = (metrics.input_tokens or 0) + (usage.get("input_tokens") or 0)
        metrics.output_tokens = (metrics.output_tokens or 0) + (usage.get("output_tokens") or 0)
        metrics.total_tokens = (metrics.total_tokens or 0) + (usage.get("total_tokens") or 0)
    metrics.model_calls = calls
    return metrics


def _merge_metrics(base: TaskMetrics, extra: TaskMetrics) -> TaskMetrics:
    def add(left: int | None, right: int | None) -> int | None:
        if left is None and right is None:
            return None
        return (left or 0) + (right or 0)

    return TaskMetrics(
        input_tokens=add(base.input_tokens, extra.input_tokens),
        output_tokens=add(base.output_tokens, extra.output_tokens),
        total_tokens=add(base.total_tokens, extra.total_tokens),
        latency_ms=base.latency_ms,
        model_calls=base.model_calls + extra.model_calls,
        retry_count=base.retry_count,
    )


def stop_reason(response: object) -> str:
    """Retourne le stopReason normalise d'une reponse Bedrock Converse."""
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("stopReason") or metadata.get("stop_reason") or ""
    return str(raw).strip().casefold()


def is_truncated_response(response: object) -> bool:
    """Indique qu'une reponse a ete coupee par la limite de tokens."""
    return stop_reason(response) in _TRUNCATION_STOP_REASONS


@dataclass
class ModelOutput:
    """Sortie consolidee d'un appel modele, reprises de troncature incluses."""

    text: str
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    truncated: bool = False
    continuations: int = 0
    tool_evidence: list[dict[str, str]] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)


def invoke_model_with_continuation(model: Any, messages: list[Any], max_continuations: int = MAX_CONTINUATIONS) -> ModelOutput:
    """Invoque un modele simple et recolle automatiquement les suites tronquees."""
    base_messages = list(messages)
    session = CURRENT_EXECUTION.get()
    call_config = {"callbacks": [UsageRecorder(session.budget, session.run_budget)]} if session else {}
    response = model.invoke(base_messages, config=call_config)
    text = extract_text(response, strip=False)
    metrics = _usage_from_response(response)
    continuations = 0
    while is_truncated_response(response) and text and continuations < max_continuations:
        continuations += 1
        response = model.invoke([*base_messages, ("ai", text), ("user", CONTINUATION_INSTRUCTION)], config=call_config)
        metrics = _merge_metrics(metrics, _usage_from_response(response))
        chunk = extract_text(response, strip=False)
        if not chunk:
            break
        text = f"{text}{chunk}"
    return ModelOutput(text=text, metrics=metrics, truncated=is_truncated_response(response), continuations=continuations)


def _output_from_messages(state_messages: list[Any]) -> ModelOutput:
    """Extract machine evidence exclusively from actual tool messages."""
    tool_evidence: list[dict[str, str]] = []
    proposal_ids: list[str] = []
    for sequence, message in enumerate(state_messages):
        if not isinstance(message, ToolMessage):
            continue
        name = str(getattr(message, "name", "") or "")
        if not name:
            continue
        content = extract_text(message)
        reference = name
        detail = content[:300]
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = None
        record: dict[str, Any] = {"sequence": sequence, "ok": False}
        if isinstance(payload, dict):
            record.update({key: payload[key] for key in (
                "ok", "applied", "content_hash", "category", "command", "checked_files", "returncode",
                "started_at_ns", "completed_at_ns", "stable_during_check"
            ) if key in payload})
            if name == "executer_commande_controlee" and payload.get("stable_during_check") is False:
                record["ok"] = False
            reference = str(
                payload.get("path")
                or payload.get("proposal_id")
                or payload.get("cwd")
                or name
            )
            proposal_id = payload.get("proposal_id")
            if payload.get("created") and proposal_id:
                proposal_ids.append(str(proposal_id))
            detail = str(payload.get("message") or payload.get("error") or "")[:300]
            if payload.get("applied") is True and payload.get("ok") is True:
                detail = (
                    "applied=true; "
                    f"backup={payload.get('backup_path') or 'none'}"
                )
            if name == "executer_commande_controlee":
                reference = str(payload.get("cwd", ".")) + ": " + " ".join(payload.get("command") or [])
                detail = (
                    f"ok={str(bool(payload.get('ok'))).lower()}; "
                    f"returncode={payload.get('returncode')}"
                )
            if name == "lire_extraits_fichiers":
                for item in payload.get("files", []):
                    if isinstance(item, dict):
                        tool_evidence.append({"kind": "file_read", "reference": item.get("path", name),
                                              "ok": item.get("ok") is True, "sequence": sequence,
                                              "content_hash": item.get("content_hash"), "detail": ""})
                continue
            if name == "appliquer_modifications_directement":
                for item in payload.get("files", []):
                    if isinstance(item, dict):
                        tool_evidence.append({"kind": "file_write", "reference": item.get("path", name),
                                              "ok": item.get("ok") is True, "applied": item.get("applied") is True,
                                              "sequence": sequence, "content_hash": item.get("content_hash"),
                                              "completed_at_ns": item.get("completed_at_ns", 0), "detail": ""})
                continue
        kind = (
            "file_write" if name.endswith("_directement")
            else "proposal" if name.startswith("proposer_")
            else "test" if name == "executer_commande_controlee" and record.get("category") == "test"
            else "command" if name == "executer_commande_controlee"
            else "file_read"
        )
        tool_evidence.append({
            **record,
            "kind": kind,
            "reference": reference,
            "detail": detail,
        })
    last = state_messages[-1] if state_messages else None
    text = extract_text(last, strip=False) if last is not None and not isinstance(last, ToolMessage) else ""
    position = len(state_messages) - 2
    while position > 0 and isinstance(state_messages[position], HumanMessage) and state_messages[position].content == CONTINUATION_INSTRUCTION:
        text = extract_text(state_messages[position - 1], strip=False) + text
        position -= 2
    # A tool result is evidence, not a generated final response.
    return ModelOutput(
        text=text,
        metrics=_usage_from_messages(state_messages),
        truncated=is_truncated_response(last),
        tool_evidence=tool_evidence,
        proposal_ids=list(dict.fromkeys(proposal_ids)),
    )


def invoke_agent_with_continuation(agent: Any, messages: list[Any], max_continuations: int = MAX_CONTINUATIONS,
                                  session: ExecutionSession | None = None, complete_result: bool = False) -> ModelOutput:
    """Resume pending graph nodes; never regenerate a tool-backed final report."""
    config: dict[str, Any] = {"recursion_limit": AGENT_RECURSION_LIMIT}
    graph_input: Any = {"messages": list(messages)}
    if session is not None:
        config["configurable"] = {"thread_id": session.key}
        config["callbacks"] = [UsageRecorder(session.budget, session.run_budget)]
        snapshot = agent.get_state(config)
        if snapshot.values:
            session.messages = snapshot.values.get("messages", [])
            graph_input = None
        for message in session.messages:
            if isinstance(message, ToolMessage):
                try:
                    payload = json.loads(extract_text(message))
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict) and payload.get("applied") is True and payload.get("path"):
                    session.writes[payload["path"]] = payload
                elif isinstance(payload, dict):
                    for item in payload.get("files", []):
                        if isinstance(item, dict) and item.get("applied") is True and item.get("path"):
                            session.writes[item["path"]] = item
                if isinstance(payload, dict):
                    for item in [payload, *payload.get("files", [])]:
                        if isinstance(item, dict) and item.get("path") and item.get("content_hash"):
                            try:
                                path = resolve_workspace_path(item["path"])
                                session.read_versions[str(path)] = item["content_hash"]
                            except ValueError:
                                pass
        if snapshot.values and not snapshot.next:
            last = session.messages[-1]
            if getattr(last, "invalid_tool_calls", []):
                graph_input = {"messages": [RemoveMessage(id=last.id), HumanMessage(content=
                    "Le dernier appel d'outil était incomplet et n'a pas été exécuté. "
                    "Régénère uniquement l'action restante ; les outils précédents ont déjà été appliqués.")]}
            elif any(not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
                     for path, digest in session.read_versions.items()):
                graph_input = {"messages": [HumanMessage(content=
                    "Des fichiers ont changé depuis l'exécution sauvegardée. Relis seulement les zones concernées "
                    "et termine la tâche sur la version actuelle, sans répéter les changements déjà présents.")]}
            elif not (complete_result and is_truncated_response(last)):
                return _output_from_messages(session.messages)
            else:
                graph_input = {"messages": [HumanMessage(content=CONTINUATION_INSTRUCTION)]}
    last_state: dict[str, Any] = {}
    for repair in range(max_continuations + 1):
        try:
            for state in agent.stream(graph_input, config=config, stream_mode="values"):
                last_state = state
                if session is not None:
                    session.messages = state.get("messages", [])
        except Exception:
            if session is not None:
                snapshot = agent.get_state(config)
                session.messages = snapshot.values.get("messages", session.messages) if snapshot.values else session.messages
            raise
        state_messages = session.messages if session is not None else last_state.get("messages", [])
        if not state_messages:
            raise ValueError("L'agent n'a produit aucun message.")
        last = state_messages[-1]
        if not getattr(last, "invalid_tool_calls", []):
            output = _output_from_messages(state_messages)
            if complete_result and output.truncated and repair < max_continuations:
                graph_input = {"messages": [HumanMessage(content=CONTINUATION_INSTRUCTION)]}
                continue
            output.continuations = repair
            # A truncated report costs no extra call. Only a missing tool
            # action merits regeneration, with the prior work retained.
            return output
        if repair == max_continuations:
            raise ValueError("Appel d'outil toujours incomplet ; les écritures précédentes sont conservées.")
        graph_input = {"messages": [RemoveMessage(id=last.id), HumanMessage(content=
            "Le dernier appel d'outil était incomplet et n'a pas été exécuté. "
            "Régénère uniquement l'action restante ; les outils précédents ont déjà été appliqués.")]}
    raise ValueError("Aucun résultat d'agent.")


def _merge_tool_evidence(text: str, output: ModelOutput, preserve_text: bool = False) -> str:
    """Ajoute les preuves réelles des outils à une enveloppe JSON valide."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if not isinstance(payload, dict):
        # Un worker peut avoir appliqué les fichiers correctement puis manquer
        # de tokens pour son bref rapport JSON. Préserve la preuve des écritures
        # et évite de classer tout le travail comme échec ou artefact inconnu.
        if not output.tool_evidence:
            return text
        writes = [
            item for item in output.tool_evidence
            if item.get("kind") == "file_write" and item.get("ok") is True and item.get("applied") is True
        ]
        payload = {
            "state": "applied" if writes else "observed",
            "summary": (
                "Modifications écrites dans le workspace."
                if writes
                else "Le worker a utilisé des outils; son rapport est incomplet."
            ),
            "references": list(dict.fromkeys(
                item.get("reference", "")
                for item in output.tool_evidence
                if item.get("reference")
            )),
            "evidence": output.tool_evidence,
            "proposal_ids": output.proposal_ids,
            "blockers": [],
            "report_incomplete": output.truncated,
        }
        if preserve_text:
            payload["result"] = text
        return json.dumps(payload, ensure_ascii=False)
    # Discard self-reported evidence. Only the runtime can populate it.
    payload["state"] = "observed"
    payload["evidence"] = output.tool_evidence
    payload["proposal_ids"] = output.proposal_ids
    payload["references"] = list(dict.fromkeys(item["reference"] for item in output.tool_evidence))
    return json.dumps(payload, ensure_ascii=False)


def _exception_name(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code:
            return str(code)
    return type(exc).__name__


def is_retryable_exception(exc: BaseException) -> bool:
    name = _exception_name(exc)
    if name in _NON_RETRYABLE_ERROR_NAMES:
        return False
    if name in _RETRYABLE_ERROR_NAMES:
        return True
    text = str(exc).casefold()
    return any(token in text for token in ("throttl", "temporar", "timeout", "timed out", "service unavailable", "connection reset"))


def _model_semaphore(model_alias: str, limit: int) -> threading.BoundedSemaphore:
    key = f"{model_alias}:{limit}"
    with _SEMAPHORE_LOCK:
        return _MODEL_SEMAPHORES.setdefault(key, threading.BoundedSemaphore(limit))


def _task_requires_reading(task: Task, context: str = "") -> bool:
    """Detecte si la tache exige une lecture reelle du workspace.

    Deux sources declenchent l'agent de lecture :
    - la tache elle-meme mentionne une lecture (titre/description) ;
    - le contexte transmis (livrables de dependances = workers subordonnes)
      contient une demande explicite de consulter un fichier precis.
    """
    haystack = f"{task.title}\n{task.description}".casefold()
    if any(token in haystack for token in _READ_INTENT_TOKENS):
        return True
    if any((WORKSPACE_ROOT / path).is_file() for path in task_paths(task)):
        return True
    context_text = str(context or "").casefold()
    return any(token in context_text for token in _READ_INTENT_TOKENS)


def _task_requires_writing(task: Task, context: str = "") -> bool:
    """Detecte si la tache exige de MODIFIER le workspace (creation de proposition).

    Comme _task_requires_reading, on inspecte le titre/description de la tache
    ainsi que le contexte transmis par les workers subordonnes.
    """
    haystack = f"{task.title}\n{task.description}".casefold()
    if any(token in haystack for token in _WRITE_INTENT_TOKENS):
        return True
    context_text = str(context or "").casefold()
    return any(token in context_text for token in _WRITE_INTENT_TOKENS)


def _task_requires_execution(task: Task) -> bool:
    text = f"{task.title}\n{task.description}".casefold()
    return any(token in text for token in (
        "pytest", "unittest", "ruff", "npm test", "npm run", "exécuter", "executer",
        "lancer les tests", "run tests", "compiler", "typecheck", "lint", "build",
    ))


def _invoke_editor_agent(model_alias: str, temperature: float | int | None, system_prompt: str, task_prompt: str) -> ModelOutput:
    """Direct editor with persistent, resumable graph state."""
    session = CURRENT_EXECUTION.get()
    agent = session.agent if session else None
    if agent is None:
        agent = get_workspace_editor_agent(model_alias=model_alias, temperature=temperature,
                                          checkpointer=session.checkpointer() if session else None)
        if session:
            session.agent = agent
    return invoke_agent_with_continuation(
        agent, [("system", system_prompt), ("user", task_prompt)], session=session,
    )


def _invoke_reader_agent(model_alias: str, temperature: float | int | None, system_prompt: str, task_prompt: str,
                         allow_execution: bool = False) -> ModelOutput:
    """Invoque un agent LangGraph de LECTURE SEULE (outils de consultation).

    Le worker peut ainsi ouvrir reellement les fichiers du workspace au lieu de
    se contenter du contexte pre-lu et tronque. L'agent n'expose aucun outil
    d'ecriture, de proposition ou de memoire.
    """
    session = CURRENT_EXECUTION.get()
    agent = session.agent if session else None
    if agent is None:
        agent = get_workspace_reader_agent(model_alias=model_alias, temperature=temperature,
                                          checkpointer=session.checkpointer() if session else None,
                                          allow_execution=allow_execution)
        if session:
            session.agent = agent
    return invoke_agent_with_continuation(
        agent, [("system", system_prompt), ("user", task_prompt)], session=session,
        complete_result=not allow_execution,
    )


def run_single_task(task: Task, context: str = "", config: WorkerExecutionConfig | None = None, cancel_event: threading.Event | None = None, progress: ProgressCallback | None = None) -> WorkerResult:
    config = config or WorkerExecutionConfig()
    started = time.time()
    result = WorkerResult(task_id=task.id, tier=task.assigned_tier, started_at=started)
    try:
        tier_conf = get_tier(task.assigned_tier)
        result.model_alias = resolve_tier_to_model_alias(task.assigned_tier)
        temperature = sanitize_temperature(result.model_alias, WORKER_TEMPERATURE)
    except Exception as exc:  # erreur de configuration, jamais retry
        result.error_type = _exception_name(exc)
        result.error = f"{result.error_type}: {exc}"
        result.completed_at = time.time()
        result.latency_ms = (result.completed_at - started) * 1000
        _emit(progress, "error", f"{task.title} : configuration invalide ({result.error}).", task.id)
        return result

    semaphore = _model_semaphore(result.model_alias, min(config.max_parallel_per_model, tier_conf.get("max_parallel", 1)))
    fingerprint = hashlib.sha256(json.dumps([task.title, task.description, task.acceptance_criteria,
                                             task.expected_output, result.model_alias, context,
                                             {key: value for key, value in config.artifacts.items() if key in task.dependencies}],
                                            ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    session = ExecutionSession(key=f"{task.execution_id}:{fingerprint}",
                               root=PROJECT_ROOT / ".agent_state", run_budget=config.budget,
                               artifacts={key: value for key, value in config.artifacts.items() if key in task.dependencies})
    runtime_token = CURRENT_EXECUTION.set(session)
    _emit(progress, "worker", f"{task.title} : en attente d'un creneau {result.model_alias}.", task.id)
    max_attempts = min(
        config.max_attempts,
        max(1, 1 + int(getattr(task, "max_retries", 1))),
    )
    for attempt in range(1, max_attempts + 1):
        result.attempt_count = attempt
        if cancel_event and cancel_event.is_set():
            result.error_type = "CancelledError"
            result.error = "Execution annulee avant l'appel au modele."
            _emit(progress, "state", f"{task.title} : annulee avant l'appel.", task.id)
            break
        try:
            with semaphore:
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Execution annulee avant l'appel au modele.")
                system_prompt = worker_system_prompt(tier_conf["label"])
                task_prompt = worker_task_prompt(
                    task.title,
                    task.description,
                    context,
                    acceptance_criteria=getattr(task, "acceptance_criteria", []),
                    expected_output=getattr(task, "expected_output", ""),
                    target_output_tokens=TARGET_OUTPUT_TOKENS,
                )
                declared_capabilities = set(getattr(task, "capabilities", []) or [])
                if {"write", "propose_write"} & declared_capabilities:
                    mode = "edition"
                elif "execute" in declared_capabilities:
                    mode = "controle"
                elif "read" in declared_capabilities:
                    mode = "controle" if _task_requires_execution(task) else "lecture"
                elif "generate" in declared_capabilities:
                    mode = "redaction"
                elif _task_requires_writing(task, context):
                    mode = "edition"
                elif _task_requires_execution(task):
                    mode = "controle"
                elif _task_requires_reading(task, context):
                    mode = "lecture"
                else:
                    mode = "redaction"

                _emit(
                    progress,
                    "worker",
                    f"{task.title} : appel {result.model_alias} en mode {mode} "
                    f"(tentative {attempt}/{max_attempts}).",
                    task.id,
                )

                if mode == "edition":
                    output = _invoke_editor_agent(result.model_alias, temperature, system_prompt, task_prompt)
                elif mode == "lecture":
                    output = _invoke_reader_agent(result.model_alias, temperature, system_prompt, task_prompt)
                elif mode == "controle":
                    output = _invoke_reader_agent(result.model_alias, temperature, system_prompt, task_prompt, allow_execution=True)
                else:
                    model = create_chat_model(model_alias=result.model_alias, temperature=temperature)
                    output = invoke_model_with_continuation(
                        model, [("system", system_prompt), ("user", task_prompt)]
                    )
            deliverable = _merge_tool_evidence(output.text.strip(), output, preserve_text=mode in {"lecture", "redaction"})
            if not deliverable:
                raise ValueError("Le modele a renvoye un livrable vide.")
            result.truncated = output.truncated
            result.tool_evidence = output.tool_evidence
            result.continuations = output.continuations
            if output.truncated:
                # Ne colle pas un marqueur en texte libre à la fin du JSON :
                # cela détruirait l'enveloppe et ferait perdre les preuves.
                try:
                    partial = json.loads(deliverable)
                except (json.JSONDecodeError, TypeError):
                    partial = None
                if not isinstance(partial, dict):
                    partial = {
                        "state": "observed",
                        "summary": (deliverable or "Sortie incomplète.")[:500],
                        "references": [],
                        "evidence": output.tool_evidence,
                        "proposal_ids": output.proposal_ids,
                        "tests": [],
                        "blockers": [],
                        "claims": [],
                        "result": deliverable if mode == "redaction" else "",
                    }
                partial["report_incomplete"] = True
                if any(item.get("kind") == "file_write" and item.get("ok") is True
                       and item.get("applied") is True for item in output.tool_evidence):
                    partial["state"] = "applied"
                deliverable = json.dumps(partial, ensure_ascii=False)
            result.deliverable = deliverable
            result.error = None
            result.error_type = None
            result.retryable = False
            result.metrics = output.metrics
            result.metrics.retry_count = attempt - 1
            result.metrics.target_output_tokens = TARGET_OUTPUT_TOKENS
            result.metrics.target_output_exceeded = bool(
                len(output.text) > TARGET_OUTPUT_TOKENS * 4
            )
            if output.truncated and mode in {"redaction", "lecture"}:
                result.error_type = "IncompleteOutput"
                result.error = "Livrable textuel incomplet après les continuations autorisées ; contenu partiel conservé."
            _emit(
                progress,
                "worker",
                f"{task.title} : livrable recu ({len(deliverable)} caracteres)."
                + (
                    f" Objectif souple de {TARGET_OUTPUT_TOKENS} tokens dépassé."
                    if result.metrics.target_output_exceeded
                    else ""
                )
                + (" Sortie tronquee." if output.truncated else ""),
                task.id,
            )
            break
        except Exception as exc:  # borne et classification centralisee
            retryable = is_retryable_exception(exc)
            result.error_type = _exception_name(exc)
            result.error = f"{result.error_type}: {exc}"
            result.retryable = retryable
            if not retryable or attempt >= max_attempts:
                _emit(progress, "error", f"{task.title} : echec ({result.error}).", task.id)
                break
            delay = min(config.max_backoff_seconds, config.base_backoff_seconds * (2 ** (attempt - 1))) + random.uniform(0.0, config.jitter_seconds)
            _emit(
                progress,
                "worker",
                f"{task.title} : erreur transitoire ({result.error_type}), "
                f"nouvelle tentative dans {delay:.1f}s.",
                task.id,
            )
            time.sleep(delay)

    if result.error and session.messages:
        partial_output = _output_from_messages(session.messages)
        result.tool_evidence = partial_output.tool_evidence
        if partial_output.tool_evidence and not result.deliverable:
            result.deliverable = _merge_tool_evidence("", partial_output)
    # The checkpoint includes messages from successful calls before a failure.
    # Preserve their usage as well; timeout billing remains explicitly unknown.
    recorded = session.budget.snapshot()
    if session is not None:
        result.metrics.input_tokens = recorded["input_tokens"]
        result.metrics.output_tokens = recorded["output_tokens"]
        result.metrics.total_tokens = recorded["input_tokens"] + recorded["output_tokens"]
        result.metrics.model_calls = recorded["model_calls"]
        result.metrics.unknown_usage_calls = recorded["unknown_usage_calls"]
        result.metrics.cache_read_tokens = recorded["cache_read_tokens"]
        result.metrics.cache_write_tokens = recorded["cache_write_tokens"]
    CURRENT_EXECUTION.reset(runtime_token)
    session.close()
    result.completed_at = time.time()
    result.latency_ms = (result.completed_at - started) * 1000
    result.metrics.latency_ms = result.latency_ms
    result.metrics.retry_count = max(0, result.attempt_count - 1)
    return result


def build_context(ledger: TaskLedger, task: Task, discovery_context: str = "") -> str:
    return compile_context(ledger, task, discovery_context)


def run_tasks_parallel(ledger: TaskLedger, tasks: list[Task], max_workers: int = MAX_PARALLEL_WORKERS, config: WorkerExecutionConfig | None = None, cancel_event: threading.Event | None = None, discovery_context: str = "", progress: ProgressCallback | None = None) -> list[WorkerResult]:
    if not tasks or ledger.cancel_requested:
        return []
    config = config or WorkerExecutionConfig(max_parallel_workers=max_workers)
    config.artifacts.update({task.id: task.artifact.result or task.deliverable or ""
                             for task in ledger.tasks if task.artifact})
    cancel_event = cancel_event or threading.Event()
    prepared: list[tuple[Task, str]] = []
    for task in tasks:
        if task not in ledger.ready_tasks():
            raise ValueError(f"La tache {task.id} n'est pas prete a etre executee.")
        attempt = ledger.begin_task(task.id)
        task.input_revision = workspace_revision()
        ledger.log_event("task_started", f"Debut de {task.title}.", task_id=task.id, attempt=attempt)
        prepared.append((task, build_context(ledger, task, discovery_context)))
    results: list[WorkerResult] = []
    workers = max(1, min(config.max_parallel_workers, len(prepared)))
    order = {task.id: index for index, (task, _) in enumerate(prepared)}
    _emit(
        progress,
        "state",
        f"Vague de {len(prepared)} tache(s), {workers} worker(s) en parallele.",
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ma-worker") as pool:
        def execute_owned(task: Task, context: str) -> WorkerResult:
            # These scheduling locks are separate from tool locks: LangGraph
            # tools run on child threads, so sharing an RLock would deadlock.
            paths = task_paths(task) if _task_requires_writing(task) or {"write", "propose_write"} & set(task.capabilities) else []
            with _SEMAPHORE_LOCK:
                locks = [_TASK_PATH_LOCKS.setdefault(path, threading.Lock()) for path in paths]
            with ExitStack() as stack:
                for lock in locks:
                    stack.enter_context(lock)
                task.input_revision = workspace_revision()
                return run_single_task(task, context, config, cancel_event, progress)
        futures = {pool.submit(execute_owned, task, context): task for task, context in prepared}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                worker_result = future.result()
                worker_result.input_revision = task.input_revision
                worker_result.output_revision = workspace_revision()
                results.append(worker_result)
            except Exception as exc:
                results.append(WorkerResult(task_id=task.id, tier=task.assigned_tier, error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__))
            completed += 1
            _emit(
                progress,
                "state",
                f"Avancement de la vague : {completed}/{len(prepared)} tache(s) terminee(s).",
                task.id,
            )
    results.sort(key=lambda result: order[result.task_id])
    return results


def apply_results(ledger: TaskLedger, results: list[WorkerResult]) -> None:
    for result in results:
        if ledger.cancel_requested and not result.ok:
            ledger.set_status(result.task_id, STATUS_CANCELLED)
            continue
        if result.ok:
            task = ledger.require_task(result.task_id)
            task.input_revision = result.input_revision
            task.output_revision = result.output_revision
            ledger.set_deliverable(result.task_id, result.deliverable, result.metrics,
                                   trusted_evidence=result.tool_evidence)
            ledger.log_event("task_succeeded", "Livrable produit.", task_id=result.task_id, model_alias=result.model_alias, attempt=result.attempt_count, metadata={"latency_ms": result.latency_ms})
        else:
            if result.deliverable and result.tool_evidence:
                task = ledger.require_task(result.task_id)
                task.input_revision = result.input_revision
                task.output_revision = result.output_revision
                ledger.set_deliverable(result.task_id, result.deliverable,
                                       trusted_evidence=result.tool_evidence)
            ledger.set_error(result.task_id, result.error or "Erreur inconnue.", result.error_type, result.retryable, result.metrics)
            ledger.log_event("task_failed", result.error or "Erreur inconnue.", level="error", task_id=result.task_id, model_alias=result.model_alias, attempt=result.attempt_count, metadata={"retryable": result.retryable, "latency_ms": result.latency_ms})
