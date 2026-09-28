import json
import unittest
import uuid

from core.multi_agent.coordination import (
    compile_context,
    parse_worker_envelope,
    repetition_ratio,
    workspace_revision,
)
from core.multi_agent.ledger import STATUS_DONE, Task, TaskLedger
from tools.custom_tools import WORKSPACE_ROOT
from tools.write_tools import (
    _register_proposal,
    approve_pending_proposals,
    clear_resolved_proposals,
    reject_proposal,
)


class CoordinationTest(unittest.TestCase):
    def test_structured_envelope_and_repetition(self):
        envelope = parse_worker_envelope(json.dumps({
            "state": "tested",
            "summary": "Tests ciblés réussis.",
            "references": ["src/a.py"],
            "evidence": [{"kind": "file_read", "reference": "src/a.py"}],
            "proposal_ids": [],
            "tests": ["pytest src/test_a.py"],
            "blockers": [],
        }))
        self.assertEqual(envelope.state, "tested")
        self.assertEqual(envelope.evidence[0].reference, "src/a.py")
        self.assertGreater(repetition_ratio("une ligne assez longue\nune ligne assez longue"), 0)

    def test_context_uses_compact_artifact_not_full_deliverable(self):
        dependency = Task("T01", "Analyser", "x", "fast")
        ledger = TaskLedger(tasks=[dependency])
        ledger.begin_task("T01")
        dependency.input_revision = workspace_revision()
        dependency.output_revision = dependency.input_revision
        ledger.set_deliverable("T01", json.dumps({
            "state": "observed",
            "summary": "Contrat identifié.",
            "references": ["api.py"],
            "evidence": [],
            "proposal_ids": [],
            "tests": [],
            "blockers": [],
        }))
        task = Task("T02", "Utiliser", "x", "fast", dependencies=["T01"])
        ledger.add_task(task)
        context = compile_context(ledger, task, "D" * 20_000)
        self.assertIn("Contrat identifié", context)
        self.assertLessEqual(len(context), 12_000)
        self.assertNotIn(dependency.deliverable, context)

    def test_applied_barrier_waits_for_artifact_state(self):
        dependency = Task("T01", "Proposer", "x", "fast")
        ledger = TaskLedger(tasks=[dependency])
        ledger.begin_task("T01")
        ledger.set_deliverable("T01", json.dumps({
            "state": "proposed",
            "summary": "Patch proposé.",
            "references": [],
            "evidence": [],
            "proposal_ids": [],
            "tests": [],
            "blockers": [],
        }))
        task = Task(
            "T02", "Vérifier", "x", "fast",
            dependencies=["T01"], dependency_state="applied",
        )
        ledger.add_task(task)
        self.assertEqual(dependency.status, STATUS_DONE)
        self.assertNotIn(task, ledger.ready_tasks())
        dependency.artifact.state = "applied"
        self.assertIn(task, ledger.ready_tasks())

    def test_group_approval_is_atomic_for_valid_files(self):
        token = uuid.uuid4().hex
        paths = [
            WORKSPACE_ROOT / f"_coordination_{token}_a.txt",
            WORKSPACE_ROOT / f"_coordination_{token}_b.txt",
        ]
        proposal_ids = []
        try:
            for index, path in enumerate(paths):
                result = json.loads(_register_proposal(
                    path=path,
                    operation="creation_complete",
                    justification="test transactionnel",
                    old_content="",
                    new_content=f"contenu {index}",
                ))
                proposal_ids.append(result["proposal_id"])
            results = approve_pending_proposals(proposal_ids)
            self.assertTrue(all(item["ok"] for item in results))
            self.assertEqual(len({item["change_set_id"] for item in results}), 1)
            self.assertEqual(paths[0].read_text(encoding="utf-8"), "contenu 0")
            self.assertEqual(paths[1].read_text(encoding="utf-8"), "contenu 1")
        finally:
            clear_resolved_proposals()
            for path in paths:
                path.unlink(missing_ok=True)

    def test_group_approval_writes_nothing_when_prevalidation_fails(self):
        token = uuid.uuid4().hex
        first = WORKSPACE_ROOT / f"_coordination_{token}_first.txt"
        conflicted = WORKSPACE_ROOT / f"_coordination_{token}_conflict.txt"
        proposal_ids = []
        try:
            for path in (first, conflicted):
                result = json.loads(_register_proposal(
                    path=path,
                    operation="creation_complete",
                    justification="test rollback",
                    old_content="",
                    new_content="proposé",
                ))
                proposal_ids.append(result["proposal_id"])
            conflicted.write_text("modification concurrente", encoding="utf-8")
            results = approve_pending_proposals(proposal_ids)
            self.assertTrue(all(not item["ok"] for item in results))
            self.assertFalse(first.exists())
            self.assertEqual(
                conflicted.read_text(encoding="utf-8"),
                "modification concurrente",
            )
        finally:
            for proposal_id in proposal_ids:
                try:
                    reject_proposal(proposal_id)
                except Exception:
                    pass
            clear_resolved_proposals()
            first.unlink(missing_ok=True)
            conflicted.unlink(missing_ok=True)

    def test_conflicting_claims_are_recorded(self):
        ledger = TaskLedger(tasks=[
            Task("T01", "Observer A", "x", "fast"),
            Task("T02", "Observer B", "x", "fast"),
        ])
        payload = {
            "state": "observed",
            "summary": "Observation.",
            "references": [],
            "evidence": [],
            "proposal_ids": [],
            "tests": [],
            "blockers": [],
            "claims": [{"subject": "route /x", "predicate": "exists", "value": "yes"}],
        }
        for task_id, value in (("T01", "yes"), ("T02", "no")):
            ledger.begin_task(task_id)
            task = ledger.require_task(task_id)
            task.output_revision = "same-revision"
            payload["claims"][0]["value"] = value
            ledger.set_deliverable(task_id, json.dumps(payload))
        self.assertEqual(len(ledger.contradictions), 1)

    def test_artifacts_and_metrics_survive_ledger_round_trip(self):
        ledger = TaskLedger(tasks=[Task("T01", "Observer", "x", "fast")])
        ledger.begin_task("T01")
        task = ledger.require_task("T01")
        task.input_revision = "before"
        task.output_revision = "after"
        ledger.set_deliverable("T01", json.dumps({
            "state": "tested",
            "summary": "Validation réussie.",
            "references": ["a.py"],
            "evidence": [{"kind": "file_read", "reference": "a.py"}],
            "proposal_ids": [],
            "tests": ["pytest"],
            "blockers": [],
            "claims": [],
        }))
        restored = TaskLedger.from_dict(ledger.to_dict())
        restored_task = restored.require_task("T01")
        self.assertEqual(restored_task.artifact.state, "tested")
        self.assertEqual(restored_task.artifact.evidence[0].reference, "a.py")
        self.assertEqual(restored_task.output_revision, "after")


if __name__ == "__main__":
    unittest.main()
