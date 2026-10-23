"""Failure, recovery, scheduling and presentation checks; no provider calls."""
import hashlib
import json
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from langgraph.prebuilt import create_react_agent
from botocore.exceptions import ConnectTimeoutError, ReadTimeoutError
from langchain_core.messages import ToolMessage

from core.agent_runtime import CURRENT_EXECUTION, ExecutionSession
from core.multi_agent.ledger import Task, TaskLedger, STATUS_DONE, STATUS_REVIEW, STATUS_TODO
from core.multi_agent.orchestrator import ChefDeProjet
from core.multi_agent.workers import run_single_task, invoke_agent_with_continuation, WorkerResult, _output_from_messages, is_retryable_exception
from tools.custom_tools import WORKSPACE_ROOT, lire_fichier, executer_commande_controlee
from tools.write_tools import DIRECT_WRITE_TOOLS
from tests.test_execution_recovery import ScriptedModel, ai, workfile
from llm_engine import compact_duplicate_tool_results


def finish(ledger, evidence):
    ledger.set_deliverable('A', '{"summary":"done","blockers":[]}', trusted_evidence=evidence)


def test_unresolved_write_blocks_dependencies_but_successful_repair_clears_it():
    ledger = TaskLedger(tasks=[Task('A','Edit','x','fast'), Task('B','Use','x','fast',dependencies=['A'])])
    ledger.begin_task('A')
    bad = {'kind':'file_write','reference':'a.py','ok':False,'sequence':1}
    finish(ledger,[bad])
    assert ledger.tasks[0].status == STATUS_REVIEW
    assert not ledger.ready_tasks()
    finish(ledger,[bad, {**bad,'ok':True,'applied':True,'sequence':2}])
    assert ledger.tasks[0].status == STATUS_DONE
    assert [t.id for t in ledger.ready_tasks()] == ['B']


def test_latest_valid_retest_survives_sync(workfile):
    path, relative = workfile
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = TaskLedger(tasks=[Task('A','Edit','x','fast')]); ledger.begin_task('A')
    write = {'kind':'file_write','reference':relative,'ok':True,'applied':True,'sequence':1}
    test = {'kind':'test','reference':'pytest','ok':True,'sequence':2,'checked_files':{relative:'old'}}
    finish(ledger,[write,test,{**write,'sequence':3},{**test,'sequence':4,'checked_files':{relative:current}}])
    ledger.sync_artifact_states()
    assert ledger.tasks[0].artifact.tested
    path.write_text('value=99\n')
    ledger.sync_artifact_states()
    assert not ledger.tasks[0].artifact.tested


def test_unscoped_success_does_not_certify_files():
    ledger=TaskLedger(tasks=[Task('A','Test','x','fast')]);ledger.begin_task('A')
    finish(ledger,[{'kind':'test','reference':'pytest','ok':True,'sequence':1,'checked_files':{}}])
    assert not ledger.tasks[0].artifact.tested


def test_commands_record_create_modify_delete_even_on_failure(workfile,tmp_path):
    path,relative=workfile
    new=path.with_name('new.py');newrel=new.relative_to(WORKSPACE_ROOT).as_posix()
    session=ExecutionSession('effects',tmp_path); token=CURRENT_EXECUTION.set(session)
    try:
        code=f"from pathlib import Path; Path({relative!r}).unlink(); Path({newrel!r}).write_text('x=3\\n'); raise RuntimeError('after effects')"
        result=json.loads(executer_commande_controlee.invoke({'commande':[sys.executable,'-c',code]}))
        assert not result['ok']
        assert {item['path'] for item in result['files']} == {relative,newrel}
        assert session.writes[relative]['deleted']
        output=_output_from_messages([ToolMessage(content=json.dumps(result),name='executer_commande_controlee',tool_call_id='c')])
        assert len([e for e in output.tool_evidence if e['kind']=='file_write']) == 2
    finally:
        CURRENT_EXECUTION.reset(token);session.close()


def test_stop_does_not_start_a_second_model_call(workfile,tmp_path):
    _,relative=workfile; stop=threading.Event()
    class CancellingModel(ScriptedModel):
        def _generate(self,*args,**kwargs):
            result=super()._generate(*args,**kwargs);stop.set();return result
    model=CancellingModel(responses=[ai(tool_calls=[{'name':'lire_fichier','args':{'chemin':relative},'id':'r','type':'tool_call'}]),ai('unexpected')])
    def factory(**kw):return create_react_agent(model,[lire_fichier],checkpointer=kw['checkpointer'])
    with patch('core.multi_agent.workers.get_workspace_reader_agent',side_effect=factory),patch('core.multi_agent.workers.PROJECT_ROOT',tmp_path):
        result=run_single_task(Task('A','Read','x','fast',capabilities=['read']),cancel_event=stop)
    assert model.position == 1
    assert result.error_type == 'ExecutionCancelled'
    assert result.metrics.model_calls == 1
    assert result.metrics.unknown_usage_calls == 0


def test_pending_resume_warns_of_changed_read(workfile,tmp_path):
    path,relative=workfile
    first=ScriptedModel(responses=[ai(tool_calls=[{'name':'lire_fichier','args':{'chemin':relative},'id':'r','type':'tool_call'}]),ConnectionError('lost')])
    session=ExecutionSession('pending',tmp_path);token=CURRENT_EXECUTION.set(session)
    try:
        graph=create_react_agent(first,[lire_fichier],checkpointer=session.checkpointer())
        with pytest.raises(ConnectionError):invoke_agent_with_continuation(graph,[('user','read')],session=session)
    finally:CURRENT_EXECUTION.reset(token);session.close()
    path.write_text('value=99\n')
    class Capture(ScriptedModel):
        seen:list=[]
        def _generate(self,messages,*args,**kwargs):
            self.seen=[str(m.content) for m in messages];return super()._generate(messages,*args,**kwargs)
    second=Capture(responses=[ai('done')]);session=ExecutionSession('pending',tmp_path);token=CURRENT_EXECUTION.set(session)
    try:
        graph=create_react_agent(second,[lire_fichier],checkpointer=session.checkpointer())
        invoke_agent_with_continuation(graph,[('user','unused')],session=session)
        assert any('Relis ces fichiers modifiés' in text for text in second.seen)
    finally:CURRENT_EXECUTION.reset(token);session.close()


def test_context_update_keeps_prior_work_in_checkpoint(tmp_path):
    class Capture(ScriptedModel):
        seen:list=[]
        def _generate(self,messages,*args,**kwargs):
            self.seen=[str(m.content) for m in messages];return super()._generate(messages,*args,**kwargs)
    model=Capture(responses=[ai('{"summary":"previous result"}'),ai('{"summary":"corrected result"}')])
    def factory(**kw):return create_react_agent(model,DIRECT_WRITE_TOOLS,checkpointer=kw['checkpointer'])
    with patch('core.multi_agent.workers.get_workspace_editor_agent',side_effect=factory),patch('core.multi_agent.workers.PROJECT_ROOT',tmp_path):
        task=Task('A','Edit','x','fast',capabilities=['write'])
        run_single_task(task,context='initial')
        cached=run_single_task(task,context='initial')
        changed=run_single_task(task,context='new correction')
    assert cached.metrics.model_calls == 0
    assert changed.ok
    assert any('previous result' in text for text in model.seen)
    assert any('new correction' in text for text in model.seen)


def test_scheduler_starts_dependency_without_waiting_for_unrelated_work():
    b_started=threading.Event()
    chief=ChefDeProjet();chief.ledger=TaskLedger(tasks=[Task('A','A','x','fast'),Task('C','C','x','fast'),Task('B','B','x','fast',dependencies=['A'])])
    def worker(task,*args,**kwargs):
        if task.id == 'B':b_started.set()
        if task.id == 'C':assert b_started.wait(3), 'B was held behind unrelated C'
        return WorkerResult(task_id=task.id,tier='fast',deliverable='{"summary":"done"}')
    with patch('core.multi_agent.workers.run_single_task',side_effect=worker),patch('core.multi_agent.workers.workspace_revision',return_value='rev'):
        chief.execute(max_parallel=2)
    assert chief.ledger.all_done()


def test_restore_releases_orphaned_tasks_without_changing_execution_id():
    ledger=TaskLedger(tasks=[Task('A','A','x','fast')]);ledger.begin_task('A');identity=ledger.tasks[0].execution_id
    restored=TaskLedger.from_dict(ledger.to_dict())
    assert restored.recover_interrupted_tasks() == ['A']
    assert restored.tasks[0].status == STATUS_TODO
    assert restored.tasks[0].execution_id == identity


def test_clear_request_builds_plan_with_one_chief_call():
    raw=json.dumps({'objective':'Read','assumptions':[], 'tasks':[{'title':'Read','description':'Read file','assigned_tier':'fast','capabilities':['read'],'depends_on':[],'acceptance_criteria':['Content read'],'expected_output':'Answer'}]})
    chief=ChefDeProjet()
    with patch.object(chief,'_ask_chief',return_value=raw) as model:
        report=chief.prepare_request('Read the file')
    assert report.task_count == 1
    assert model.call_count == 1


def test_necessary_question_does_not_build_a_plan():
    chief=ChefDeProjet()
    with patch.object(chief,'_ask_chief',return_value='{"question":"Quel destinataire ?"}') as model:
        assert chief.prepare_request('Prepare a message') is None
    assert not chief.ledger.tasks
    assert chief.history[-1][1] == 'Quel destinataire ?'
    assert model.call_count == 1


def test_known_control_uses_no_model_and_reuses_unchanged_result(workfile,tmp_path):
    task=Task('A','Check','Check version','fast',capabilities=['execute'],command=[sys.executable,'--version'])
    with patch('core.multi_agent.workers.PROJECT_ROOT',tmp_path),patch('core.multi_agent.workers.create_chat_model') as model:
        result=run_single_task(task)
        with patch('tools.custom_tools.subprocess.run',side_effect=AssertionError('replayed')):
            cached=run_single_task(task)
    assert result.ok and cached.ok
    assert result.metrics.model_calls == 0
    model.assert_not_called()


def test_machine_proofs_are_kept_out_of_model_input():
    message=ToolMessage(name='executer_commande_controlee',tool_call_id='c',content=json.dumps({'ok':True,'checked_files':{'a.py':'hash'},'stdout':'1 passed'}))
    result=compact_duplicate_tool_results({'messages':[message]})['llm_input_messages'][0]
    assert 'checked_files' not in json.loads(result.content)
    assert 'checked_files' in json.loads(message.content)
    assert '1 passed' in result.content


def test_task_ui_shows_summary_instead_of_raw_envelope():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('''
from core.multi_agent.ui import _render_task_board
from core.multi_agent.orchestrator import ChefDeProjet
from core.multi_agent.ledger import Task
chief=ChefDeProjet()
chief.ledger.add_task(Task('A','Corriger','Corriger le fichier','fast'))
chief.ledger.begin_task('A')
chief.ledger.set_deliverable('A','{"summary":"Correction enregistrée","blockers":[]}')
_render_task_board(chief)
''').run()
    assert not app.exception
    assert any(item.value == 'Correction enregistrée' for item in app.markdown)
    assert not any('"blockers"' in item.value for item in app.markdown)


def test_ambiguous_read_timeout_is_not_retried_but_connect_timeout_is():
    assert not is_retryable_exception(ReadTimeoutError(endpoint_url="https://bedrock.invalid"))
    assert is_retryable_exception(ConnectTimeoutError(endpoint_url="https://bedrock.invalid"))


def test_replan_feedback_is_saved_before_model_call():
    chief = ChefDeProjet()
    saves = []
    chief.set_save_callback(lambda: saves.append(True))
    chief.apply_feedback("Garde les tâches déjà terminées.")
    assert saves == [True]
    assert "Garde les tâches déjà terminées." in chief.history[-1][1]
