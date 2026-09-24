"""Execution en arriere-plan des operations longues du mode multi-agent.

Streamlit relance le script a chaque interaction. Une operation longue executee
directement dans le flux du script (appel Bedrock, vague de workers) est donc
interrompue des que l'utilisateur touche un widget, change d'onglet ou que le
navigateur se reconnecte.

Ce module deporte ces operations dans un thread dedie et expose leur avancement
via une file d'evenements. Le script Streamlit devient un simple observateur :
il draine la file, affiche la progression, et ne bloque jamais.

Contrainte respectee : AUCUN import de streamlit ici. Le thread de travail ne
doit jamais toucher `st.session_state`, qui n'existe que dans le contexte du
script principal.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"

FINISHED_STATUSES = {STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED}

# Au-dela, les evenements les plus anciens sont oublies : un chantier long
# produit des centaines de lignes et seul le fil recent est utile a l'ecran.
MAX_RETAINED_EVENTS = 300


@dataclass
class ProgressEvent:
    """Evenement d'avancement emis par le thread de travail."""

    kind: str
    message: str
    task_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class BackgroundJob:
    """Operation longue executee dans un thread, observable sans blocage.

    Cycle de vie : `idle` -> `running` -> `done` / `error` / `cancelled`.

    L'objet est stocke tel quel dans `st.session_state`. C'est volontaire :
    Streamlit conserve les objets Python de la session dans le meme processus,
    donc le thread et le script partagent bien la meme instance.
    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.status = STATUS_IDLE
        self.result: Any = None
        self.error: Optional[str] = None
        self.traceback: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.cancel_event = threading.Event()
        self._queue: "queue.Queue[ProgressEvent]" = queue.Queue()
        self._events: list[ProgressEvent] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Cote thread de travail
    # ------------------------------------------------------------------

    def emit(self, kind: str, message: str, task_id: str | None = None) -> None:
        """Publie un evenement d'avancement. Appelable depuis n'importe quel thread."""
        self._queue.put(ProgressEvent(kind=kind, message=message, task_id=task_id))

    def start(self, target: Callable[["BackgroundJob"], Any]) -> None:
        """Lance `target(job)` dans un thread demon.

        Le thread est demon afin qu'un arret de Streamlit ne reste jamais
        bloque sur une operation en cours.
        """
        with self._lock:
            if self.status == STATUS_RUNNING:
                raise RuntimeError(
                    f"L'operation {self.operation!r} est deja en cours."
                )
            self.status = STATUS_RUNNING
            self.started_at = time.time()
            self.finished_at = None
            self.result = None
            self.error = None
            self.traceback = None

        def runner() -> None:
            try:
                result = target(self)
            except Exception as exc:  # toute exception est capturee et exposee
                with self._lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.traceback = traceback.format_exc()
                    self.status = (
                        STATUS_CANCELLED
                        if self.cancel_event.is_set()
                        else STATUS_ERROR
                    )
                    self.finished_at = time.time()
                self.emit("error", self.error or "Erreur inconnue.")
                return
            with self._lock:
                self.result = result
                self.status = (
                    STATUS_CANCELLED if self.cancel_event.is_set() else STATUS_DONE
                )
                self.finished_at = time.time()
            self.emit("state", f"Operation {self.operation} terminee.")

        self._thread = threading.Thread(
            target=runner,
            name=f"ma-job-{self.operation}",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Cote script Streamlit
    # ------------------------------------------------------------------

    def drain(self) -> list[ProgressEvent]:
        """Transfere les evenements en attente vers l'historique et le retourne.

        Retourne l'historique complet conserve (borne a MAX_RETAINED_EVENTS),
        afin que l'affichage reste stable entre deux reruns Streamlit.
        """
        while True:
            try:
                self._events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if len(self._events) > MAX_RETAINED_EVENTS:
            self._events = self._events[-MAX_RETAINED_EVENTS:]
        return list(self._events)

    @property
    def events(self) -> list[ProgressEvent]:
        return list(self._events)

    def is_running(self) -> bool:
        return self.status == STATUS_RUNNING

    def is_finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def request_cancel(self) -> None:
        """Demande l'arret cooperatif de l'operation.

        L'arret n'est pas immediat : un appel Bedrock deja parti va jusqu'a son
        terme. Les vagues suivantes ne sont pas lancees.
        """
        self.cancel_event.set()
        self.emit("state", "Interruption demandee, arret apres l'etape en cours.")

    def join(self, timeout: float | None = None) -> None:
        """Attend la fin du thread. Utilise par les tests, jamais par l'UI."""
        if self._thread is not None:
            self._thread.join(timeout)
