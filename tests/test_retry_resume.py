"""Explicit retries must advance partial terminal checkpoints without replay."""
from unittest.mock import patch

from langgraph.prebuilt import create_react_agent

from core.multi_agent.ledger import Task, TaskLedger, STATUS_REVIEW, STATUS_DONE
from core.multi_agent.workers import run_single_task, apply_results
from tests.test_execution_recovery import ScriptedModel, ai, workfile
from tools.write_tools import DIRECT_WRITE_TOOLS


def test_retry_advances_partial_checkpoint_once_and_keeps_applied_writes(workfile, tmp_path):
    path, relative = workfile
    def write(old, new, identity):
        return ai(tool_calls=[{'name': 'remplacer_bloc_directement', 'args': {
            'chemin': relative, 'ancien_texte': old, 'nouveau_texte': new},
            'id': identity, 'type': 'tool_call'}])
    model = ScriptedModel(responses=[
        write('value = 1', 'value = 2', 'first'),
        ai('{"summary":"Partiel", "blockers":["Il reste une modification."]}'),
        write('value = 2', 'value = 3', 'remaining'),
        ai('{"summary":"Terminé", "blockers":[]}'),
    ])
    def factory(**kwargs):
        return create_react_agent(model, DIRECT_WRITE_TOOLS, checkpointer=kwargs['checkpointer'])
    task = Task('A', 'Modifier', 'Terminer les modifications.', 'fast', capabilities=['write'])
    ledger = TaskLedger(tasks=[task])
    original_execution = task.execution_id
    with patch('core.multi_agent.workers.get_workspace_editor_agent', side_effect=factory), \
         patch('core.multi_agent.workers.PROJECT_ROOT', tmp_path):
        ledger.begin_task('A')
        apply_results(ledger, [run_single_task(task)])
        assert task.status == STATUS_REVIEW
        assert path.read_text() == 'value = 2\n'
        ledger.retry_task('A')
        ledger.begin_task('A')
        resumed = run_single_task(task)
        apply_results(ledger, [resumed])
        assert task.status == STATUS_DONE
        assert path.read_text() == 'value = 3\n'
        assert task.execution_id == original_execution
        assert model.position == 4
        assert len([e for e in resumed.tool_evidence if e['kind'] == 'file_write' and e.get('applied')]) == 2
        cached = run_single_task(task)
        assert cached.metrics.model_calls == 0
        assert model.position == 4
