"""Targeted checks for the observed expensive run; no AWS calls."""
import json
import os
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from core.agent_runtime import RunBudget, UsageRecorder
from core.multi_agent.ledger import Task, TaskLedger
from core.multi_agent.orchestrator import ChefDeProjet
from core.multi_agent.workers import run_single_task, apply_results
from llm_engine import create_chat_model, get_workspace_reader_agent, compact_duplicate_tool_results
from tests.test_execution_recovery import ScriptedModel, ai, workfile
from tools.custom_tools import lire_fichier
from tools.write_tools import DIRECT_WRITE_TOOLS


class BedrockStub:
    def __init__(self):
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        return {'output': {'message': {'role': 'assistant', 'content': [{'text': 'done'}]}},
                'stopReason': 'end_turn', 'usage': {'inputTokens': 10, 'outputTokens': 1, 'totalTokens': 11},
                'metrics': {'latencyMs': 1}, 'ResponseMetadata': {'HTTPStatusCode': 200}}


def test_agent_cache_markers_survive_binding_tools():
    client = BedrockStub()
    with patch.dict(os.environ, {}, clear=True), patch('llm_engine.create_bedrock_runtime_client', return_value=client), patch('langchain_aws.chat_models.bedrock_converse.create_aws_client', return_value=client):
        agent = get_workspace_reader_agent(model_alias='claude-sonnet-5.5')
        agent.invoke({'messages': [SystemMessage(content='instructions'), HumanMessage(content='read')]})
    request = client.requests[0]
    assert request['system'][-1] == {'cachePoint': {'type': 'default'}}
    assert request['toolConfig']['tools'][-1] == {'cachePoint': {'type': 'default'}}
    assert request['messages'][-1]['content'][-1] == {'cachePoint': {'type': 'default'}}


def test_single_chief_call_stays_uncached_and_empty_override_disables_agent_cache():
    client = BedrockStub()
    with patch.dict(os.environ, {}, clear=True), patch('llm_engine.create_bedrock_runtime_client', return_value=client), patch('langchain_aws.chat_models.bedrock_converse.create_aws_client', return_value=client):
        chief = create_chat_model('claude-sonnet-5.5')
        assert not getattr(chief, 'kwargs', {}).get('cache_control')
    with patch.dict(os.environ, {'BEDROCK_PROMPT_CACHE_MODELS': ''}, clear=True), patch('llm_engine.create_bedrock_runtime_client', return_value=client), patch('langchain_aws.chat_models.bedrock_converse.create_aws_client', return_value=client):
        agent_model = create_chat_model('claude-sonnet-5.5', cache_for_agent=True)
        assert not getattr(agent_model, 'kwargs', {}).get('cache_control')


def test_cache_projection_does_not_rewrite_old_prefix_when_duplicate_is_added():
    first = ToolMessage(content='a' * 1000, name='lire_fichier', tool_call_id='first')
    duplicate = ToolMessage(content=first.content, name=first.name, tool_call_id='again')
    projected = compact_duplicate_tool_results({'messages': [first, duplicate]}, preserve_prefix=True)['llm_input_messages']
    assert projected[0].content == first.content
    assert projected[1].content == duplicate.content


def test_editor_without_writes_is_not_done_and_keeps_its_generated_code(workfile, tmp_path):
    _, relative = workfile
    end = ai('value = 77\n')
    end.response_metadata = {'stopReason': 'max_tokens'}
    model = ScriptedModel(responses=[ai(tool_calls=[{'name': 'lire_fichier', 'args': {'chemin': relative}, 'id': 'read', 'type': 'tool_call'}]), end,
                                     ai(tool_calls=[{'name': 'remplacer_bloc_directement', 'args': {'chemin': relative, 'ancien_texte': 'value = 1', 'nouveau_texte': 'value = 77'}, 'id': 'write', 'type': 'tool_call'}]), ai('{"summary":"Appliqué","blockers":[]}')])
    def factory(**kwargs):
        return create_react_agent(model, [lire_fichier, *DIRECT_WRITE_TOOLS], checkpointer=kwargs['checkpointer'])
    task = Task('A', 'Edit', 'Modifier.', 'fast', capabilities=['write'])
    ledger = TaskLedger(tasks=[task]); ledger.begin_task('A')
    with patch('core.multi_agent.workers.get_workspace_editor_agent', side_effect=factory), patch('core.multi_agent.workers.PROJECT_ROOT', tmp_path):
        result = run_single_task(task)
        apply_results(ledger, [result])
        assert result.error_type == 'IncompleteOutput'
        assert task.status != 'termine'
        assert task.artifact.result == 'value = 77\n'
        assert model.position == 2
        chief = ChefDeProjet(); chief.ledger = ledger
        with patch.object(chief, '_ask_chief', side_effect=AssertionError('paid synthesis')):
            assert 'aucune écriture prouvée' in chief.synthesize()
        ledger.retry_task('A'); ledger.begin_task('A')
        resumed = run_single_task(task)
        apply_results(ledger, [resumed])
    assert task.status == 'termine'
    assert model.position == 4
    assert task.artifact.state == 'applied'


def test_structured_success_without_an_edit_is_kept_for_review():
    ledger = TaskLedger(tasks=[Task('A', 'Edit', 'Modify', 'fast', capabilities=['write'])])
    ledger.begin_task('A')
    ledger.set_deliverable('A', '{"summary":"done","blockers":[]}', trusted_evidence=[])
    assert ledger.tasks[0].status == 'a revoir'


def test_cache_write_details_with_ttl_are_counted_without_double_counting():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import LLMResult, ChatGeneration
    for details in [{'ephemeral_5m_input_tokens': 100}, {'cache_creation': 100, 'ephemeral_5m_input_tokens': 100}]:
        budget = RunBudget()
        message = AIMessage(content='ok', usage_metadata={'input_tokens': 100, 'output_tokens': 1, 'total_tokens': 101, 'input_token_details': details})
        UsageRecorder(budget).on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        assert budget.cache_write_tokens == 100


def test_old_run_with_observed_edit_is_reconciled_on_restore():
    ledger = TaskLedger(tasks=[Task('A', 'Edit', 'Modify', 'fast', capabilities=['write'])])
    ledger.begin_task('A')
    ledger.set_deliverable('A', '{"summary":"done","blockers":[]}', trusted_evidence=[])
    old = ledger.to_dict()
    old['tasks'][0]['status'] = 'termine'
    old['tasks'][0]['artifact']['blockers'] = []
    restored = TaskLedger.from_dict(old)
    assert restored.tasks[0].status == 'a revoir'
    assert restored.tasks[0].artifact.blockers
