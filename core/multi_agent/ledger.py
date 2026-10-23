"""Task Ledger pur Python, source de verite du systeme multi-agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from core.multi_agent.coordination import (
    Artifact,
    Evidence,
    STATE_RANK,
    parse_worker_envelope,
    repetition_ratio,
    workspace_revision,
)

STATE_INTERVIEW = "INTERVIEW"
STATE_DISCOVERY = "DISCOVERY"
STATE_PLANIF = "PLANIF"
STATE_EXEC = "EXEC"
STATE_REVIEW = "REVIEW"
STATE_SYNTHESE = "SYNTHESE"
STATE_CANCELLED = "CANCELLED"
VALID_STATES = {STATE_INTERVIEW, STATE_DISCOVERY, STATE_PLANIF, STATE_EXEC, STATE_REVIEW, STATE_SYNTHESE, STATE_CANCELLED}

STATUS_TODO = "a faire"
STATUS_DOING = "en cours"
STATUS_DONE = "termine"
STATUS_FAILED = "echec"
STATUS_BLOCKED = "bloque"
STATUS_REVIEW = "a revoir"
STATUS_CANCELLED = "annule"
VALID_TASK_STATUSES = {STATUS_TODO, STATUS_DOING, STATUS_DONE, STATUS_FAILED, STATUS_BLOCKED, STATUS_REVIEW, STATUS_CANCELLED}
TERMINAL_TASK_STATUSES = {STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskMetrics:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    model_calls: int = 0
    retry_count: int = 0
    estimated_cost: float | None = None
    currency: str | None = None
    target_output_tokens: int | None = None
    target_output_exceeded: bool = False
    repetition_ratio: float = 0.0
    files_read: int = 0
    artifacts_validated: int = 0
    tokens_per_validated_artifact: float | None = None


@dataclass
class TaskAttempt:
    number: int
    started_at: str
    completed_at: str | None = None
    status: str = STATUS_DOING
    error_type: str | None = None
    error: str | None = None
    retryable: bool = False
    latency_ms: float | None = None


@dataclass
class Event:
    kind: str
    message: str
    level: str = "info"
    run_id: str = ""
    task_id: str | None = None
    model_alias: str | None = None
    attempt: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass
class Task:
    id: str
    title: str
    description: str
    assigned_tier: str
    assigned_role: str = "worker"
    capabilities: list[str] = field(default_factory=list)
    max_retries: int = 1
    max_escalations: int = 1
    # Revue croisee : quand requires_review vaut True, le livrable est soumis
    # a un agent critique d'un tier superieur avant la synthese.
    requires_review: bool = False
    review_verdict: str | None = None
    review_comment: str = ""
    status: str = STATUS_TODO
    dependencies: list[str] = field(default_factory=list)
    dependency_state: str = "completed"
    deliverable: str | None = None
    artifact: Artifact | None = None
    input_revision: str = ""
    output_revision: str = ""
    notes: str = ""
    error: str | None = None
    error_type: str | None = None
    retryable: bool = False
    attempt_count: int = 0
    attempts: list[TaskAttempt] = field(default_factory=list)
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Task":
        data = dict(raw)
        data.setdefault("capabilities", [])
        data.setdefault("max_retries", 1)
        data.setdefault("max_escalations", 1)
        data.setdefault("dependency_state", "completed")
        data["metrics"] = TaskMetrics(**data.get("metrics", {}))
        data["attempts"] = [TaskAttempt(**item) for item in data.get("attempts", [])]
        artifact = data.get("artifact")
        if isinstance(artifact, dict):
            artifact = dict(artifact)
            artifact["evidence"] = [
                Evidence(**item) for item in artifact.get("evidence", [])
            ]
            data["artifact"] = Artifact(**artifact)
        return cls(**data)


@dataclass
class TaskLedger:
    objective: str = ""
    state: str = STATE_INTERVIEW
    tasks: list[Task] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    final_answer: str | None = None
    run_id: str = field(default_factory=lambda: uuid4().hex)
    events: list[Event] = field(default_factory=list)
    cancel_requested: bool = False
    cancelled_at: str | None = None
    cancellation_reason: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    chief_history: list[tuple[str, str]] = field(default_factory=list)
    discovery_context: str = ""
    discovered_files: list[str] = field(default_factory=list)
    discovery_errors: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in VALID_STATES:
            raise ValueError(f"Etat invalide : {self.state}")

    def get_task(self, task_id: str) -> Task | None:
        return next((task for task in self.tasks if task.id == task_id), None)

    def require_task(self, task_id: str) -> Task:
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Tache inconnue : {task_id}")
        return task

    def is_done(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        return task is not None and task.status == STATUS_DONE

    def ready_tasks(self) -> list[Task]:
        if self.cancel_requested:
            return []
        ready: list[Task] = []
        for task in self.tasks:
            if task.status != STATUS_TODO:
                continue
            dependencies_ready = True
            for dependency_id in task.dependencies:
                dependency = self.get_task(dependency_id)
                if dependency is None or dependency.status != STATUS_DONE:
                    dependencies_ready = False
                    break
                if task.dependency_state != "completed":
                    state = dependency.artifact.state if dependency.artifact else "observed"
                    if STATE_RANK.get(state, -1) < STATE_RANK.get(task.dependency_state, 0):
                        dependencies_ready = False
                        break
            if dependencies_ready:
                ready.append(task)
        return ready

    def all_done(self) -> bool:
        return bool(self.tasks) and all(task.status == STATUS_DONE for task in self.tasks)

    def stalled_tasks(self) -> list[tuple[Task, str]]:
        """Taches TODO qui ne peuvent plus devenir executables sans action humaine.

        Deux causes : une barriere d'etat qu'un artefact dependant n'atteint
        pas (proposition non appliquee), ou une dependance definitivement
        non terminee. Sans ce diagnostic, ces taches bloquent indefiniment la
        synthese alors qu'aucune progression automatique n'est possible.
        """
        ready_ids = {task.id for task in self.ready_tasks()}
        stalled: list[tuple[Task, str]] = []
        for task in self.tasks:
            if task.status != STATUS_TODO or task.id in ready_ids:
                continue
            reasons: list[str] = []
            for dependency_id in task.dependencies:
                dependency = self.get_task(dependency_id)
                if dependency is None:
                    reasons.append(f"dependance introuvable : {dependency_id}")
                    continue
                if dependency.status != STATUS_DONE:
                    reasons.append(
                        f"« {dependency.title} » est en statut {dependency.status}"
                    )
                    continue
                if task.dependency_state != "completed":
                    state = dependency.artifact.state if dependency.artifact else "observed"
                    if STATE_RANK.get(state, -1) < STATE_RANK.get(task.dependency_state, 0):
                        reasons.append(
                            f"« {dependency.title} » est a l'etat '{state}' alors que "
                            f"'{task.dependency_state}' est requis"
                        )
            if reasons:
                stalled.append((task, " ; ".join(reasons)))
        return stalled

    def barrier_blocked_tasks(self) -> list[tuple[Task, str]]:
        """Taches bloquees uniquement par une barriere d'etat d'artefact."""
        blocked: list[tuple[Task, str]] = []
        for task, reason in self.stalled_tasks():
            if task.dependency_state == "completed":
                continue
            if all(
                (dependency := self.get_task(dependency_id)) is not None
                and dependency.status == STATUS_DONE
                for dependency_id in task.dependencies
            ):
                blocked.append((task, reason))
        return blocked

    def release_barriers(self, task_ids: list[str] | None = None) -> list[str]:
        """Ramene la barriere a 'completed' pour reprendre l'execution.

        Utilise quand un artefact dependant reste a l'etat propose : la
        decision d'avancer malgre tout appartient a l'humain, et elle est
        tracee comme une decision du chantier.
        """
        targets = (
            [task for task, _ in self.barrier_blocked_tasks()]
            if task_ids is None
            else [task for task in self.tasks if task.id in set(task_ids)]
        )
        released: list[str] = []
        for task in targets:
            if task.dependency_state == "completed":
                continue
            previous = task.dependency_state
            task.dependency_state = "completed"
            task.updated_at = utc_now_iso()
            released.append(task.id)
            self.decisions.append(
                f"Barriere levee pour « {task.title} » : '{previous}' -> 'completed'."
            )
        if released:
            self.updated_at = utc_now_iso()
        return released

    def sync_artifact_states(self) -> None:
        """Réconcilie les artefacts proposés avec l'état réel des propositions."""
        proposal_ids = [
            proposal_id
            for task in self.tasks if task.artifact
            for proposal_id in task.artifact.proposal_ids
        ]
        if not proposal_ids:
            return
        from tools.write_tools import get_proposal_statuses

        statuses = get_proposal_statuses(proposal_ids)
        for task in self.tasks:
            artifact = task.artifact
            if not artifact or not artifact.proposal_ids:
                continue
            current = [statuses.get(item, "unknown") for item in artifact.proposal_ids]
            if current and all(item == "approved" for item in current):
                artifact.state = "applied"
                task.output_revision = workspace_revision()
                artifact.output_revision = task.output_revision

    def pending_count(self) -> int:
        return sum(task.status in {STATUS_TODO, STATUS_DOING, STATUS_FAILED, STATUS_BLOCKED, STATUS_REVIEW} for task in self.tasks)

    def tasks_with_status(self, *statuses: str) -> list[Task]:
        return [task for task in self.tasks if task.status in statuses]

    def add_task(self, task: Task) -> None:
        if self.get_task(task.id):
            raise ValueError(f"Tache deja presente : {task.id}")
        unknown = set(task.dependencies) - {item.id for item in self.tasks}
        # Les dependances vers des taches ajoutees plus tard restent permises ici.
        if task.status not in VALID_TASK_STATUSES:
            raise ValueError(f"Statut invalide : {task.status}")
        self.tasks.append(task)
        self.updated_at = utc_now_iso()

    def set_state(self, state: str) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"Etat invalide : {state}")
        self.state = state
        self.updated_at = utc_now_iso()

    def set_status(self, task_id: str, status: str) -> None:
        if status not in VALID_TASK_STATUSES:
            raise ValueError(f"Statut invalide : {status}")
        task = self.require_task(task_id)
        task.status = status
        task.updated_at = utc_now_iso()
        self.updated_at = task.updated_at

    def begin_task(self, task_id: str) -> int:
        task = self.require_task(task_id)
        if task.status not in {STATUS_TODO, STATUS_FAILED, STATUS_REVIEW}:
            raise ValueError(f"La tache {task_id} ne peut pas demarrer depuis {task.status}.")
        task.attempt_count += 1
        task.attempts.append(TaskAttempt(number=task.attempt_count, started_at=utc_now_iso()))
        self.set_status(task_id, STATUS_DOING)
        return task.attempt_count

    def set_deliverable(self, task_id: str, content: str, metrics: TaskMetrics | None = None) -> None:
        if not content or not content.strip():
            raise ValueError("Un livrable termine ne peut pas etre vide.")
        task = self.require_task(task_id)
        task.deliverable = content.strip()
        task.error = None
        task.error_type = None
        task.retryable = False
        if metrics is not None:
            task.metrics = metrics
        envelope = parse_worker_envelope(content)
        successful_test = any(
            item.kind == "test"
            and ("ok=true" in item.detail.casefold() or "returncode=0" in item.detail.casefold())
            for item in envelope.evidence
        )
        successful_write = any(
            item.kind == "file_write"
            and "applied=true" in item.detail.casefold()
            for item in envelope.evidence
        )
        artifact_state = envelope.state
        if successful_write and STATE_RANK.get(artifact_state, -1) < STATE_RANK["applied"]:
            artifact_state = "applied"
        if successful_test and artifact_state not in {"proposed", "approved"}:
            artifact_state = "tested"
        task.artifact = Artifact(
            id=f"{task.id}:{task.attempt_count}",
            task_id=task.id,
            kind="worker_result",
            state=artifact_state,
            summary=envelope.summary,
            references=envelope.references,
            evidence=envelope.evidence,
            proposal_ids=envelope.proposal_ids,
            input_revision=task.input_revision,
            output_revision=task.output_revision,
            tested=bool(envelope.tests) or successful_test
            or STATE_RANK[artifact_state] >= STATE_RANK["tested"],
            claims=envelope.claims,
        )
        self._detect_claim_contradictions(task)
        task.metrics.repetition_ratio = repetition_ratio(content)
        task.metrics.files_read = len({
            item.reference for item in envelope.evidence if item.kind in {"file", "file_read"}
        })
        task.metrics.artifacts_validated = int(
            STATE_RANK[artifact_state] >= STATE_RANK["tested"]
        )
        if task.metrics.artifacts_validated and task.metrics.total_tokens:
            task.metrics.tokens_per_validated_artifact = (
                task.metrics.total_tokens / task.metrics.artifacts_validated
            )
        if task.attempts:
            task.attempts[-1].status = STATUS_DONE
            task.attempts[-1].completed_at = utc_now_iso()
            task.attempts[-1].latency_ms = task.metrics.latency_ms
        self.set_status(task_id, STATUS_DONE)
        self.refresh_blocked_tasks()

    def _detect_claim_contradictions(self, current_task: Task) -> None:
        """Signale les affirmations incompatibles sur une même révision."""
        current = current_task.artifact
        if not current or not current.claims:
            return
        for other_task in self.tasks:
            other = other_task.artifact
            if (
                other_task.id == current_task.id
                or not other
                or not other.claims
                or other.output_revision != current.output_revision
            ):
                continue
            previous = {
                (
                    item.get("subject", "").casefold(),
                    item.get("predicate", "").casefold(),
                ): item.get("value", "")
                for item in other.claims
            }
            for claim in current.claims:
                key = (
                    claim.get("subject", "").casefold(),
                    claim.get("predicate", "").casefold(),
                )
                old_value = previous.get(key)
                new_value = claim.get("value", "")
                if old_value is None or old_value.casefold() == new_value.casefold():
                    continue
                if any(
                    item.get("claim_key") == list(key)
                    and not item.get("resolved", False)
                    for item in self.contradictions
                ):
                    continue
                self.contradictions.append({
                    "id": f"C{len(self.contradictions) + 1:03d}",
                    "task_ids": [other_task.id, current_task.id],
                    "claim_key": list(key),
                    "message": (
                        f"{claim.get('subject')} / {claim.get('predicate')} : "
                        f"{old_value!r} contre {new_value!r} sur la même révision."
                    ),
                    "resolved": False,
                })

    def set_error(self, task_id: str, error: str, error_type: str | None = None, retryable: bool = False, metrics: TaskMetrics | None = None) -> None:
        task = self.require_task(task_id)
        task.error = error
        task.error_type = error_type
        task.retryable = retryable
        task.notes = (task.notes + f"\n[ERREUR] {error}").strip()
        if metrics is not None:
            task.metrics = metrics
        if task.attempts:
            attempt = task.attempts[-1]
            attempt.status = STATUS_FAILED
            attempt.completed_at = utc_now_iso()
            attempt.error = error
            attempt.error_type = error_type
            attempt.retryable = retryable
            attempt.latency_ms = task.metrics.latency_ms
        self.set_status(task_id, STATUS_FAILED)
        self.refresh_blocked_tasks()

    def retry_task(self, task_id: str) -> None:
        if self.cancel_requested:
            raise ValueError("Impossible de relancer une tache : le chantier est annule.")
        task = self.require_task(task_id)
        if task.status not in {STATUS_FAILED, STATUS_CANCELLED, STATUS_REVIEW, STATUS_BLOCKED}:
            raise ValueError(f"La tache {task_id} n'est pas relancable depuis {task.status}.")
        if not all(self.is_done(dep) for dep in task.dependencies):
            self.set_status(task_id, STATUS_BLOCKED)
            raise ValueError(f"La tache {task_id} reste bloquee par ses dependances.")
        task.error = None
        task.error_type = None
        task.retryable = False
        # Un nouveau livrable doit etre recritique : on oublie l'ancien verdict.
        task.review_verdict = None
        task.review_comment = ""
        self.set_status(task_id, STATUS_TODO)
        self.log_event("task_retry", f"Relance demandee pour {task.title}.", task_id=task_id)

    def escalate_tier(self, task_id: str) -> bool:
        """Promeut le tier de la tache vers le palier superieur assignable.

        Retourne True si une escalade a eu lieu, False si la tache est deja au
        palier maximum. Utilise par la relance intelligente d'une tache ratee.
        """
        from core.multi_agent.roles import TIER_ESCALATION_ORDER

        task = self.require_task(task_id)
        order = TIER_ESCALATION_ORDER
        try:
            position = order.index(task.assigned_tier)
        except ValueError:
            position = 0
        if position >= len(order) - 1:
            return False
        previous = task.assigned_tier
        task.assigned_tier = order[position + 1]
        task.updated_at = utc_now_iso()
        self.updated_at = task.updated_at
        self.log_event(
            "tier_escalated",
            f"Tier de {task.title} promu de {previous} vers {task.assigned_tier}.",
            task_id=task_id,
        )
        return True

    def report_failure(self, task_id: str, reason: str) -> None:
        """Consigne un echec juge non trivial dans les remarques de la tache.

        Le worker/orchestrateur appelle ceci quand la solution n'est PAS
        evidente : au lieu de relancer aveuglement, le probleme est signale
        dans les notes pour la revue humaine et la synthese.
        """
        message = str(reason or "").strip() or "Cause non precisee."
        task = self.require_task(task_id)
        task.notes = (task.notes + f"\n[A SIGNALER] {message}").strip()
        task.updated_at = utc_now_iso()
        self.updated_at = task.updated_at
        self.log_event(
            "failure_reported",
            f"Echec signale pour revue sur {task.title} : {message}",
            level="warning",
            task_id=task_id,
        )

    def refresh_blocked_tasks(self) -> None:
        for task in self.tasks:
            deps = [self.get_task(dep) for dep in task.dependencies]
            failed_dependency = any(dep and dep.status in {STATUS_FAILED, STATUS_CANCELLED} for dep in deps)
            if task.status == STATUS_TODO and failed_dependency:
                self.set_status(task.id, STATUS_BLOCKED)
            elif task.status == STATUS_BLOCKED and deps and all(dep and dep.status == STATUS_DONE for dep in deps):
                self.set_status(task.id, STATUS_TODO)

    def request_cancellation(self, reason: str | None = None) -> None:
        self.cancel_requested = True
        self.cancelled_at = utc_now_iso()
        self.cancellation_reason = reason
        self.state = STATE_CANCELLED
        for task in self.tasks:
            if task.status in {STATUS_TODO, STATUS_BLOCKED, STATUS_REVIEW}:
                task.status = STATUS_CANCELLED
                task.updated_at = self.cancelled_at
        self.log_event("cancel_requested", reason or "Annulation demandee.", level="warning")

    def log_decision(self, decision: str) -> None:
        if decision.strip():
            self.decisions.append(decision.strip())
            self.updated_at = utc_now_iso()

    def log_event(self, kind: str, message: str, level: str = "info", task_id: str | None = None, model_alias: str | None = None, attempt: int | None = None, metadata: dict[str, Any] | None = None) -> Event:
        event = Event(kind=kind, message=message, level=level, run_id=self.run_id, task_id=task_id, model_alias=model_alias, attempt=attempt, metadata=metadata or {})
        self.events.append(event)
        self.updated_at = event.timestamp
        return event

    def team_structure(self) -> dict[str, list[Task]]:
        grouped: dict[str, list[Task]] = {}
        for task in self.tasks:
            grouped.setdefault(task.assigned_tier, []).append(task)
        return grouped

    def metrics_summary(self) -> dict[str, Any]:
        known = [task.metrics for task in self.tasks]
        def total(field_name: str) -> int:
            return sum(getattr(item, field_name) or 0 for item in known)
        artifacts = sum(item.artifacts_validated for item in known)
        total_tokens = total("total_tokens")
        repetition = (
            sum(item.repetition_ratio for item in known) / len(known)
            if known else 0.0
        )
        return {"task_count": len(self.tasks), "done": len(self.tasks_with_status(STATUS_DONE)), "failed": len(self.tasks_with_status(STATUS_FAILED)), "blocked": len(self.tasks_with_status(STATUS_BLOCKED)), "cancelled": len(self.tasks_with_status(STATUS_CANCELLED)), "model_calls": total("model_calls"), "retries": total("retry_count"), "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"), "total_tokens": total_tokens, "files_read": total("files_read"), "artifacts_validated": artifacts, "tokens_per_validated_artifact": (total_tokens / artifacts if artifacts else None), "average_repetition_ratio": repetition}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskLedger":
        payload = dict(data)
        payload["tasks"] = [Task.from_dict(item) for item in payload.get("tasks", [])]
        payload["events"] = [Event(**item) for item in payload.get("events", [])]
        return cls(**payload)
