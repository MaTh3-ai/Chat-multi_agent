"""Task Ledger pur Python, source de verite du systeme multi-agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

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
    deliverable: str | None = None
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
        data["metrics"] = TaskMetrics(**data.get("metrics", {}))
        data["attempts"] = [TaskAttempt(**item) for item in data.get("attempts", [])]
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
        return [task for task in self.tasks if task.status == STATUS_TODO and all(self.is_done(dep) for dep in task.dependencies)]

    def all_done(self) -> bool:
        return bool(self.tasks) and all(task.status == STATUS_DONE for task in self.tasks)

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
        if task.attempts:
            task.attempts[-1].status = STATUS_DONE
            task.attempts[-1].completed_at = utc_now_iso()
            task.attempts[-1].latency_ms = task.metrics.latency_ms
        self.set_status(task_id, STATUS_DONE)
        self.refresh_blocked_tasks()

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
        return {"task_count": len(self.tasks), "done": len(self.tasks_with_status(STATUS_DONE)), "failed": len(self.tasks_with_status(STATUS_FAILED)), "blocked": len(self.tasks_with_status(STATUS_BLOCKED)), "cancelled": len(self.tasks_with_status(STATUS_CANCELLED)), "model_calls": total("model_calls"), "retries": total("retry_count"), "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"), "total_tokens": total("total_tokens")}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskLedger":
        payload = dict(data)
        payload["tasks"] = [Task.from_dict(item) for item in payload.get("tasks", [])]
        payload["events"] = [Event(**item) for item in payload.get("events", [])]
        return cls(**payload)
