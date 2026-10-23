"""Routing and coordination regressions, without Bedrock requests."""
import json
from unittest.mock import patch

from core.multi_agent.ledger import Task, TaskLedger
from core.multi_agent.orchestrator import ChefDeProjet
from core.multi_agent.planning import build_ledger
from core.multi_agent.coordination import Artifact


def test_work_kind_routes_without_classifier_and_preserves_legacy_override():
    kinds = ['extraction', 'routine', 'complex', 'critical']
    raw = {'objective': 'Work', 'tasks': [
        {'title': kind, 'description': 'Work', 'work_kind': kind} for kind in kinds]}
    ledger, _ = build_ledger(json.dumps(raw))
    assert [t.assigned_tier for t in ledger.tasks] == ['nano', 'fast', 'smart', 'expert']
    raw['tasks'][1]['assigned_tier'] = 'expert'
    ledger, _ = build_ledger(json.dumps(raw))
    assert ledger.tasks[1].assigned_tier == 'expert'
    assert TaskLedger.from_dict(ledger.to_dict()).tasks[0].assigned_tier == 'nano'


def test_chief_routes_current_request_and_respects_explicit_configuration():
    chief = ChefDeProjet()
    with patch.dict('os.environ', {'MULTI_AGENT_CHIEF_TIER': ''}):
        assert chief._chief_tier() == 'fast'
        chief.history = [('user', 'Revoir la sécurité des données')]
        assert chief._chief_tier() == 'smart'
        assert chief._chief_tier('expert') == 'expert'
    with patch.dict('os.environ', {'MULTI_AGENT_CHIEF_TIER': 'nano'}):
        assert chief._chief_tier() == 'nano'


def test_many_routine_tasks_do_not_force_opus():
    chief = ChefDeProjet()
    chief.ledger.tasks = [Task(str(i), 'Documentation', 'Update', 'fast') for i in range(6)]
    with patch.dict('os.environ', {'MULTI_AGENT_CHIEF_TIER': ''}):
        assert chief._chief_tier() == 'fast'
        chief.ledger.tasks[0].assigned_tier = 'smart'
        assert chief._chief_tier() == 'smart'


def test_code_delivery_with_free_text_does_not_pay_for_rewrite():
    chief = ChefDeProjet()
    task = Task('A', 'Modification', 'Edit', 'smart', capabilities=['write'])
    chief.ledger.tasks = [task]
    chief.ledger.begin_task('A')
    chief.ledger.set_deliverable('A', '{"summary":"Travail partiel", "result":"Détails", "blockers":["Reste à finir"]}', trusted_evidence=[])
    with patch.object(chief, '_ask_chief', side_effect=AssertionError('Unnecessary synthesis')):
        assert 'Travail partiel' in chief.synthesize()


def test_textual_multi_task_synthesis_uses_haiku_and_conflicts_use_sonnet():
    chief = ChefDeProjet()
    for i in range(2):
        task = Task(str(i), 'Analyse', 'Read', 'smart', capabilities=['read'])
        chief.ledger.add_task(task)
        chief.ledger.begin_task(task.id)
        chief.ledger.set_deliverable(task.id, json.dumps({'summary': 'Observed', 'result': 'Answer', 'blockers': []}), trusted_evidence=[])
    with patch.object(chief, '_ask_chief', return_value='Answer') as ask:
        chief.synthesize()
        assert ask.call_args.kwargs['tier'] == 'fast'
        chief.ledger.contradictions = [{'id': 'C', 'resolved': False, 'message': 'Conflict'}]
        chief.synthesize()
        assert ask.call_args.kwargs['tier'] == 'smart'
