import unittest

from core.multi_agent.ledger import Task, TaskLedger
from core.multi_agent.plan_preview import (
    _apply_edits,
    _enrich_description,
    _validate_dependencies,
)
from core.multi_agent.planning import build_ledger, estimate_plan_cost
from core.multi_agent.roles import worker_task_prompt


def make_ledger() -> TaskLedger:
    return TaskLedger(
        tasks=[
            Task(
                id="T01",
                title="Analyser",
                description="Analyser le besoin.\n\nSortie attendue : rapport\n\n"
                "Critères d'acceptation :\n- rapport lisible",
                assigned_tier="smart",
            ),
            Task(
                id="T02",
                title="Produire",
                description="Produire le résultat.",
                assigned_tier="fast",
                dependencies=["T01"],
            ),
        ]
    )


class PlanPreviewTest(unittest.TestCase):
    def test_plan_defaults_capabilities_and_budgets(self):
        raw = '{"objective":"x","tasks":[' \
            '{"title":"a","description":"a","assigned_tier":"fast"},' \
            '{"title":"b","description":"b","assigned_tier":"fast"},' \
            '{"title":"c","description":"c","assigned_tier":"fast"}]}'
        ledger, report = build_ledger(raw)
        self.assertEqual(ledger.tasks[0].capabilities, ["auto"])
        self.assertEqual(ledger.tasks[0].max_retries, 1)
        self.assertEqual(ledger.tasks[0].max_escalations, 1)
        self.assertGreater(report.estimated_cost_units, 0)
        self.assertEqual(report.estimated_cost_units, estimate_plan_cost(ledger))

    def test_enrich_description_matches_execution_shape(self):
        self.assertEqual(
            _enrich_description("Faire", "un rapport", ["complet"], True),
            "Faire\n\nSortie attendue : un rapport\n\n"
            "Critères d'acceptation :\n- complet\n\n"
            "Revue humaine ou critique ciblée requise avant synthèse.",
        )

    def test_cycle_is_rejected_before_mutation(self):
        ledger = make_ledger()
        edits = {
            "T01": {"dependencies": ["T02"]},
            "T02": {"dependencies": ["T01"]},
        }
        self.assertIn("Cycle", _validate_dependencies(ledger, edits))
        with self.assertRaisesRegex(ValueError, "Cycle"):
            _apply_edits(ledger, edits)
        self.assertEqual(ledger.tasks[0].dependencies, [])

    def test_unknown_dependency_is_rejected(self):
        ledger = make_ledger()
        edits = {"T01": {"dependencies": ["T99"]}}
        with self.assertRaisesRegex(ValueError, "inconnue"):
            _apply_edits(ledger, edits)

    def test_apply_edits_uses_canonical_description(self):
        ledger = make_ledger()
        edits = {
            "T01": {
                "title": "Analyser mieux",
                "description": "Lire les éléments.",
                "expected_output": "une analyse",
                "criteria": ["faits vérifiables"],
                "tier": "smart",
                "requires_review": True,
                "dependencies": [],
            }
        }
        _apply_edits(ledger, edits)
        self.assertEqual(
            ledger.tasks[0].description,
            _enrich_description(
                "Lire les éléments.",
                "une analyse",
                ["faits vérifiables"],
                True,
            ),
        )

    def test_worker_output_target_is_explicitly_non_blocking(self):
        prompt = worker_task_prompt(
            "Analyser",
            "Terminer l'analyse.",
            target_output_tokens=1200,
        )
        self.assertIn("Vise environ 1200 tokens", prompt)
        self.assertIn("Ce n'est jamais une limite", prompt)
        self.assertIn("termine la tâche", prompt)


if __name__ == "__main__":
    unittest.main()
