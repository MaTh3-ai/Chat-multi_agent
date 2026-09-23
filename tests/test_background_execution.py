"""Tests de l'execution en arriere-plan du mode multi-agent.

Bibliotheque standard uniquement, aucun appel Bedrock : les modeles sont
remplaces par des doubles de test.

Execution :
    python -m unittest tests.test_background_execution -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.multi_agent import workers
from core.multi_agent.background import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    BackgroundJob,
)
from core.multi_agent.ledger import (
    STATUS_DONE as TASK_DONE,
    STATUS_TODO,
    Task,
    TaskLedger,
)
from core.multi_agent.orchestrator import ChefDeProjet


class FakeResponse:
    """Reponse minimale compatible avec extract_text et les metriques."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = text
        self.response_metadata = {"stopReason": stop_reason}
        self.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }


class FakeModel:
    def __init__(self, text: str = "Livrable de test.", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return FakeResponse(self.text)


class BackgroundJobTest(unittest.TestCase):
    def test_job_execute_et_expose_le_resultat(self):
        job = BackgroundJob("test")

        def target(current_job):
            current_job.emit("state", "etape 1")
            current_job.emit("worker", "etape 2")
            return {"valeur": 42}

        job.start(target)
        job.join(timeout=5)

        self.assertEqual(job.status, STATUS_DONE)
        self.assertEqual(job.result, {"valeur": 42})
        self.assertIsNone(job.error)
        messages = [event.message for event in job.drain()]
        self.assertIn("etape 1", messages)
        self.assertIn("etape 2", messages)

    def test_job_capture_les_exceptions_sans_les_propager(self):
        job = BackgroundJob("test")

        def target(_):
            raise ValueError("defaillance simulee")

        job.start(target)
        job.join(timeout=5)

        self.assertEqual(job.status, STATUS_ERROR)
        self.assertIn("defaillance simulee", job.error or "")
        self.assertIn("ValueError", job.traceback or "")

    def test_annulation_cooperative(self):
        job = BackgroundJob("test")
        entered = threading.Event()

        def target(current_job):
            entered.set()
            while not current_job.cancel_event.is_set():
                time.sleep(0.01)
            return {}

        job.start(target)
        self.assertTrue(entered.wait(timeout=5))
        job.request_cancel()
        job.join(timeout=5)

        self.assertEqual(job.status, STATUS_CANCELLED)

    def test_drain_conserve_l_historique_entre_deux_appels(self):
        job = BackgroundJob("test")
        job.emit("state", "premier")
        self.assertEqual(len(job.drain()), 1)
        job.emit("state", "second")
        events = job.drain()
        self.assertEqual([event.message for event in events], ["premier", "second"])

    def test_job_ne_peut_pas_etre_demarre_deux_fois(self):
        job = BackgroundJob("test")
        release = threading.Event()
        job.start(lambda _: release.wait(timeout=5))
        with self.assertRaises(RuntimeError):
            job.start(lambda _: None)
        release.set()
        job.join(timeout=5)


class WorkerProgressTest(unittest.TestCase):
    def setUp(self):
        self._original_model = workers.create_chat_model
        workers.create_chat_model = lambda **kwargs: FakeModel()

    def tearDown(self):
        workers.create_chat_model = self._original_model

    def _ledger(self) -> TaskLedger:
        ledger = TaskLedger(objective="Test")
        ledger.add_task(
            Task(
                id="T1",
                title="Rediger une note",
                description="Produire une note de synthese.",
                assigned_tier="fast",
            )
        )
        return ledger

    def test_run_tasks_parallel_publie_la_progression(self):
        ledger = self._ledger()
        events: list[tuple[str, str, str | None]] = []

        results = workers.run_tasks_parallel(
            ledger,
            ledger.ready_tasks(),
            max_workers=1,
            progress=lambda kind, message, task_id: events.append((kind, message, task_id)),
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok, results[0].error)
        self.assertTrue(events, "aucun evenement de progression publie")
        messages = " | ".join(message for _, message, _ in events)
        self.assertIn("Rediger une note", messages)
        self.assertIn("Avancement de la vague", messages)

    def test_un_observateur_defaillant_n_interrompt_pas_le_worker(self):
        ledger = self._ledger()

        def broken(kind, message, task_id):
            raise RuntimeError("observateur casse")

        results = workers.run_tasks_parallel(
            ledger, ledger.ready_tasks(), max_workers=1, progress=broken
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok, results[0].error)

    def test_annulation_avant_appel_modele(self):
        ledger = self._ledger()
        cancel = threading.Event()
        cancel.set()

        results = workers.run_tasks_parallel(
            ledger, ledger.ready_tasks(), max_workers=1, cancel_event=cancel
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].error_type, "CancelledError")


class ChefProgressTest(unittest.TestCase):
    def setUp(self):
        self._original_model = workers.create_chat_model
        workers.create_chat_model = lambda **kwargs: FakeModel()

    def tearDown(self):
        workers.create_chat_model = self._original_model

    def _chief(self) -> ChefDeProjet:
        chief = ChefDeProjet()
        chief.ledger = TaskLedger(objective="Test")
        chief.ledger.add_task(
            Task(
                id="T1",
                title="Rediger",
                description="Produire un texte.",
                assigned_tier="fast",
            )
        )
        return chief

    def test_log_est_relaye_vers_l_observateur(self):
        chief = ChefDeProjet()
        recues: list[tuple[str, str]] = []
        chief.set_progress_callback(
            lambda kind, message, task_id: recues.append((kind, message))
        )

        chief.log("state", "message de test")

        self.assertIn(("state", "message de test"), recues)

    def test_observateur_defaillant_n_interrompt_pas_le_chef(self):
        chief = ChefDeProjet()

        def broken(kind, message, task_id):
            raise RuntimeError("casse")

        chief.set_progress_callback(broken)
        chief.log("state", "message")  # ne doit pas lever

    def test_execute_interrompu_avant_toute_vague(self):
        chief = self._chief()
        cancel = threading.Event()
        cancel.set()

        chief.execute(cancel_event=cancel)

        self.assertEqual(chief.ledger.get_task("T1").status, STATUS_TODO)
        messages = [event.message for event in chief.events]
        self.assertTrue(
            any("interrompue" in message.lower() for message in messages),
            f"interruption non journalisee : {messages}",
        )

    def test_execute_complet_produit_les_livrables(self):
        chief = self._chief()
        chief.execute()

        task = chief.ledger.get_task("T1")
        self.assertEqual(task.status, TASK_DONE)
        self.assertTrue(task.deliverable)

    def test_execute_dans_un_job_publie_la_progression(self):
        chief = self._chief()
        job = BackgroundJob("execution")
        chief.set_progress_callback(job.emit)

        job.start(lambda current_job: chief.execute(cancel_event=current_job.cancel_event))
        job.join(timeout=30)

        self.assertEqual(job.status, STATUS_DONE, job.traceback)
        messages = " | ".join(event.message for event in job.drain())
        self.assertIn("Vague 1", messages)
        self.assertIn("Rediger", messages)
        self.assertEqual(chief.ledger.get_task("T1").status, TASK_DONE)

    def test_sauvegarde_declenchee_apres_chaque_vague(self):
        chief = self._chief()
        appels: list[int] = []
        chief.set_save_callback(lambda: appels.append(1))

        chief.execute()

        self.assertTrue(appels, "aucune sauvegarde declenchee pendant l'execution")


if __name__ == "__main__":
    unittest.main()
