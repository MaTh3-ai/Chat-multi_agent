import json

from core.multi_agent.coordination import parse_worker_envelope
from core.multi_agent.planning import build_ledger, estimate_plan_cost
from core.multi_agent.roles import worker_task_prompt, worker_system_prompt
from core.multi_agent.workers import ModelOutput, _merge_tool_evidence
from core.models import get_tier


def test_worker_prompt_includes_acceptance_contract():
    prompt = worker_task_prompt(
        "Corriger le chargement",
        "Corriger le bug dans le module.",
        acceptance_criteria=["Le module se charge sans exception."],
        expected_output="Un correctif directement appliqué.",
    )

    assert "Le module se charge sans exception." in prompt
    assert "Un correctif directement appliqué." in prompt


def test_simple_plan_can_use_one_task_and_cost_estimate_counts_nested_retries():
    raw_plan = json.dumps({
        "objective": "Corriger un bug simple.",
        "assumptions": [],
        "tasks": [{
            "title": "Corriger le bug",
            "description": "Appliquer le correctif ciblé.",
            "assigned_tier": "fast",
            "capabilities": ["write"],
            "max_retries": 1,
            "max_escalations": 1,
            "depends_on": [],
            "dependency_state": "completed",
            "acceptance_criteria": ["Le correctif est présent."],
            "expected_output": "Le fichier corrigé.",
            "requires_review": False,
        }],
    })

    ledger, report = build_ledger(raw_plan)

    assert report.task_count == 1
    assert len(ledger.tasks) == 1
    assert ledger.tasks[0].capabilities == ["write"]
    assert estimate_plan_cost(ledger) == get_tier("fast")["relative_cost"] * 4


def test_truncated_report_preserves_successful_direct_write_evidence():
    output = ModelOutput(
        text="",
        truncated=True,
        tool_evidence=[{
            "kind": "file_write",
            "reference": "src/module.py",
            "detail": "applied=true; backup=backups/module.py.bak",
        }],
        proposal_ids=["proposal-1"],
    )

    serialized = _merge_tool_evidence("réponse coupée", output)
    envelope = parse_worker_envelope(serialized)

    assert envelope.state == "applied"
    assert "src/module.py" in envelope.references
    assert envelope.proposal_ids == ["proposal-1"]
    assert envelope.blockers


def test_worker_prompt_allows_direct_edits_and_reports_truthfully():
    prompt = worker_system_prompt("Rapide")

    assert "modifie le workspace toi-même" in prompt
    assert "appliqué, non testé" in prompt
