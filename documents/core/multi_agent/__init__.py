"""Package multi-agent : orchestrateur hiérarchique + task ledger partagé.

Pattern : Orchestrator-Worker.
- Un CHEF DE PROJET (tier expert) est le seul interlocuteur de l'utilisateur.
- Il découpe l'objectif en tâches, forme l'équipe et route le travail.
- Les WORKERS (tiers nano/fast/smart) exécutent une tâche chacun.
- Toute la communication passe par un TASK LEDGER partagé (étoile),
  jamais directement entre workers.

Les briques sans dépendance LLM (ledger, roles) sont testables isolément.
"""

from core.multi_agent.ledger import (
    Task,
    TaskLedger,
    STATE_INTERVIEW,
    STATE_PLANIF,
    STATE_EXEC,
    STATE_SYNTHESE,
    STATUS_TODO,
    STATUS_DOING,
    STATUS_DONE,
    STATUS_REVIEW,
)
from core.multi_agent.roles import (
    chief_system_prompt,
    worker_system_prompt,
    PLAN_JSON_SCHEMA,
    ASSIGNABLE_TIERS,
)

__all__ = [
    "Task",
    "TaskLedger",
    "STATE_INTERVIEW",
    "STATE_PLANIF",
    "STATE_EXEC",
    "STATE_SYNTHESE",
    "STATUS_TODO",
    "STATUS_DOING",
    "STATUS_DONE",
    "STATUS_REVIEW",
    "chief_system_prompt",
    "worker_system_prompt",
    "PLAN_JSON_SCHEMA",
    "ASSIGNABLE_TIERS",
]
