"""Regression scenarios: no live model, AWS credentials or paid calls."""
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt import create_react_agent

from core.agent_runtime import CURRENT_EXECUTION, ExecutionBudgetExceeded, ExecutionSession, RunBudget, command_category
from core.multi_agent.ledger import Task, TaskLedger, TaskMetrics
from core.multi_agent.workers import (
    ModelOutput, WorkerExecutionConfig, _merge_tool_evidence, _output_from_messages,
    apply_results, invoke_agent_with_continuation, run_single_task,
)
from core.multi_agent.orchestrator import ChefDeProjet
from tools.custom_tools import WORKSPACE_ROOT, READ_ONLY_TOOLS, lire_fichier, lire_extraits_fichiers, consulter_diff
from tools.write_tools import DIRECT_WRITE_TOOLS, appliquer_modifications_directement, remplacer_bloc_directement
from llm_engine import guard_truncated_tool_calls, compact_duplicate_tool_results


class ScriptedModel(BaseChatModel):
    responses: list
    position: int = 0

    @property
    def _llm_type(self):
        return "scripted-no-provider"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.position]
        self.position += 1
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])


def ai(content="", tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls or [],
                     usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


@pytest.fixture
def workfile():
    with tempfile.TemporaryDirectory(prefix="execution_", dir=WORKSPACE_ROOT) as directory:
        path = Path(directory) / "sample.py"
        path.write_text("value = 1\n")
        yield path, path.relative_to(WORKSPACE_ROOT).as_posix()


def test_transient_error_resumes_after_write_and_clears_error(workfile, tmp_path):
    path, relative = workfile
    model = ScriptedModel(responses=[
        ai(tool_calls=[{"name": "remplacer_bloc_directement", "args": {
            "chemin": relative, "ancien_texte": "value = 1", "nouveau_texte": "value = 2"
        }, "id": "write-1", "type": "tool_call"}]),
        ConnectionError("response lost"),
        ai('{"summary":"Valeur corrigée.","blockers":[]}'),
    ])
    def factory(**kwargs):
        return create_react_agent(model, DIRECT_WRITE_TOOLS, checkpointer=kwargs["checkpointer"])
    with patch("core.multi_agent.workers.get_workspace_editor_agent", side_effect=factory), \
         patch("core.multi_agent.workers.PROJECT_ROOT", tmp_path):
        task = Task("T1", "Corriger", "Corriger la valeur.", "fast", capabilities=["write"])
        result = run_single_task(task, config=WorkerExecutionConfig(base_backoff_seconds=0, jitter_seconds=0))
        cached = run_single_task(task)
    assert result.ok
    assert result.error is None
    assert result.attempt_count == 2
    assert path.read_text() == "value = 2\n"
    assert model.position == 3  # no replay of the successful model/tool nodes
    assert len([e for e in result.tool_evidence if e["kind"] == "file_write"]) == 1
    assert result.metrics.model_calls == 3
    assert result.metrics.total_tokens == 30
    assert result.metrics.unknown_usage_calls == 1
    assert cached.ok
    assert cached.metrics.model_calls == 0


def test_sqlite_checkpoint_survives_session_restart(workfile, tmp_path):
    path, relative = workfile
    first = ScriptedModel(responses=[
        ai(tool_calls=[{"name": "remplacer_bloc_directement", "args": {
            "chemin": relative, "ancien_texte": "value = 1", "nouveau_texte": "value = 3"
        }, "id": "write-persistent", "type": "tool_call"}]), ConnectionError("lost")])
    session = ExecutionSession("restart", tmp_path)
    token = CURRENT_EXECUTION.set(session)
    try:
        graph = create_react_agent(first, DIRECT_WRITE_TOOLS, checkpointer=session.checkpointer())
        with pytest.raises(ConnectionError):
            invoke_agent_with_continuation(graph, [("user", "edit")], session=session)
    finally:
        CURRENT_EXECUTION.reset(token)
        session.close()
    assert path.read_text() == "value = 3\n"
    second = ScriptedModel(responses=[ai('{"summary":"Terminé."}')])
    restarted = ExecutionSession("restart", tmp_path)
    token = CURRENT_EXECUTION.set(restarted)
    try:
        graph = create_react_agent(second, DIRECT_WRITE_TOOLS, checkpointer=restarted.checkpointer())
        output = invoke_agent_with_continuation(graph, [("user", "must not restart")], session=restarted)
        assert second.position == 1
        assert len([e for e in output.tool_evidence if e["kind"] == "file_write"]) == 1
    finally:
        CURRENT_EXECUTION.reset(token)
        restarted.close()


def test_failed_write_is_not_applied_and_model_cannot_forge_evidence():
    output = _output_from_messages([
        ToolMessage(name="remplacer_bloc_directement", tool_call_id="bad",
                    content=json.dumps({"ok": False, "applied": False, "path": "missing.py"})),
        ai('{"state":"verified","summary":"done","evidence":[{"kind":"test","reference":"fake"}]}')
    ])
    serialized = _merge_tool_evidence(output.text, output)
    ledger = TaskLedger(tasks=[Task("T1", "Edit", "x", "fast")])
    ledger.begin_task("T1")
    ledger.set_deliverable("T1", serialized, trusted_evidence=output.tool_evidence)
    artifact = ledger.tasks[0].artifact
    assert artifact.state == "observed"
    assert not artifact.tested
    assert len(artifact.evidence) == 1
    assert artifact.evidence[0].reference == "missing.py"


@pytest.mark.parametrize("command, category", [
    (["python", "--version"], "command"), (["python", "-c", "print('pytest')"], "command"),
    (["pytest", "--collect-only"], "command"), (["python", "-m", "pytest", "--version"], "command"),
    (["python", "-m", "pytest", "tests/test_api.py"], "test"),
    (["npm", "run", "test"], "test"), (["ruff", "check", "."], "lint"),
])
def test_only_real_test_invocations_are_classified_as_tests(command, category):
    assert command_category(command) == category


def test_test_before_write_or_for_stale_content_does_not_validate(workfile):
    path, relative = workfile
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = TaskLedger(tasks=[Task("T1", "Edit", "x", "fast")])
    ledger.begin_task("T1")
    write = {"kind": "file_write", "reference": relative, "ok": True, "applied": True, "sequence": 2}
    test = {"kind": "test", "reference": "pytest", "ok": True, "sequence": 1, "checked_files": {relative: digest}}
    ledger.set_deliverable("T1", '{"summary":"done"}', trusted_evidence=[test, write])
    assert ledger.tasks[0].artifact.state == "applied"
    test["sequence"] = 3
    ledger.set_deliverable("T1", '{"summary":"done"}', trusted_evidence=[write, test])
    assert ledger.tasks[0].artifact.state == "tested"
    path.write_text("value = 9\n")
    ledger.set_deliverable("T1", '{"summary":"done"}', trusted_evidence=[write, test])
    assert ledger.tasks[0].artifact.state == "applied"


def test_budget_stop_preserves_write_and_returns_partial_artifact(workfile, tmp_path):
    path, relative = workfile
    model = ScriptedModel(responses=[ai(tool_calls=[{"name": "remplacer_bloc_directement", "args": {
        "chemin": relative, "ancien_texte": "value = 1", "nouveau_texte": "value = 4"
    }, "id": "budget-write", "type": "tool_call"}]), ai("must not be called")])
    def factory(**kwargs):
        return create_react_agent(model, DIRECT_WRITE_TOOLS, checkpointer=kwargs["checkpointer"])
    with patch("core.multi_agent.workers.get_workspace_editor_agent", side_effect=factory), \
         patch("core.multi_agent.workers.PROJECT_ROOT", tmp_path):
        task = Task("T1", "Edit", "Modifier.", "fast", capabilities=["write"])
        ledger = TaskLedger(tasks=[task])
        ledger.begin_task("T1")
        result = run_single_task(task, config=WorkerExecutionConfig(budget=RunBudget(max_calls=1)))
        apply_results(ledger, [result])
    assert model.position == 1
    assert path.read_text() == "value = 4\n"
    assert not result.ok
    assert ledger.tasks[0].artifact.state == "applied"
    assert ledger.tasks[0].error


def test_stale_read_rejected_and_repeated_success_not_reapplied(workfile, tmp_path):
    path, relative = workfile
    session = ExecutionSession("safe-write", tmp_path)
    token = CURRENT_EXECUTION.set(session)
    try:
        read = json.loads(lire_fichier.invoke({"chemin": relative}))
        assert read["ok"]
        path.write_text("value = 7\n")
        args = {"chemin": relative, "ancien_texte": "value = 7", "nouveau_texte": "value = 8"}
        refused = json.loads(remplacer_bloc_directement.invoke(args))
        assert not refused["applied"]
        lire_fichier.invoke({"chemin": relative})
        written = json.loads(remplacer_bloc_directement.invoke(args))
        assert written["applied"]
        replay = json.loads(remplacer_bloc_directement.invoke(args))
        assert replay["applied"] and replay["replayed"]
        assert path.read_text() == "value = 8\n"
        assert json.loads(consulter_diff.invoke({"chemin": relative}))["ok"]
    finally:
        CURRENT_EXECUTION.reset(token)
        session.close()


def test_batch_prevalidation_prevents_partial_changes(workfile):
    path, relative = workfile
    second = path.with_name("second.py")
    second_relative = second.relative_to(WORKSPACE_ROOT).as_posix()
    result = json.loads(appliquer_modifications_directement.invoke({"modifications": [
        {"chemin": relative, "ancien_texte": "value = 1", "nouveau_texte": "value = 2"},
        {"chemin": second_relative, "contenu": "def invalid(:\n"},
    ]}))
    assert not result["ok"]
    assert path.read_text() == "value = 1\n"
    assert not second.exists()


def test_backend_summary_does_not_pay_for_an_extra_synthesis(workfile):
    _path, relative = workfile
    chief = ChefDeProjet()
    chief.ledger = TaskLedger(tasks=[Task("T1", "Edit", "x", "fast")])
    chief.ledger.begin_task("T1")
    chief.ledger.set_deliverable("T1", '{"summary":"Correction appliquée."}', trusted_evidence=[
        {"kind": "file_write", "reference": relative, "ok": True, "applied": True, "sequence": 1}])
    with patch.object(chief, "_ask_chief", side_effect=AssertionError("unnecessary synthesis")):
        final = chief.synthesize()
    assert relative in final
    assert "Aucun test exécuté" in final


def test_metrics_accumulate_across_attempts_and_survive_reload():
    ledger = TaskLedger(tasks=[Task("T1", "Edit", "x", "fast")])
    ledger.begin_task("T1")
    ledger.set_error("T1", "lost", retryable=True, metrics=TaskMetrics(total_tokens=12, model_calls=2))
    ledger.retry_task("T1")
    ledger.begin_task("T1")
    ledger.set_deliverable("T1", '{"summary":"done"}', TaskMetrics(total_tokens=8, model_calls=1))
    restored = TaskLedger.from_dict(ledger.to_dict())
    assert restored.metrics_summary()["total_tokens"] == 20
    assert restored.metrics_summary()["model_calls"] == 3


def test_batch_reads_return_line_numbers_and_hash(workfile):
    _path, relative = workfile
    result = json.loads(lire_extraits_fichiers.invoke({"extraits": [{"chemin": relative, "debut_ligne": 1, "fin_ligne": 1}]}))
    assert result["files"][0]["content"] == "1: value = 1"
    assert result["files"][0]["content_hash"]


def test_truncated_tool_call_is_replaced_without_executing_partial_code(workfile, tmp_path):
    path, relative = workfile
    cut = ai(tool_calls=[{"name": "remplacer_bloc_directement", "args": {
        "chemin": relative, "ancien_texte": "value = 1", "nouveau_texte": "value = 99"
    }, "id": "cut", "type": "tool_call"}])
    cut.response_metadata = {"stopReason": "max_tokens"}
    complete = ai(tool_calls=[{"name": "remplacer_bloc_directement", "args": {
        "chemin": relative, "ancien_texte": "value = 1", "nouveau_texte": "value = 2"
    }, "id": "complete", "type": "tool_call"}])
    model = ScriptedModel(responses=[cut, complete, ai('{"summary":"Corrigé."}')])
    session = ExecutionSession("truncated-tool", tmp_path)
    token = CURRENT_EXECUTION.set(session)
    try:
        graph = create_react_agent(model, DIRECT_WRITE_TOOLS, checkpointer=session.checkpointer(),
                                   post_model_hook=guard_truncated_tool_calls)
        output = invoke_agent_with_continuation(graph, [("user", "edit")], session=session)
        assert path.read_text() == "value = 2\n"
        assert output.continuations == 1
        assert len([e for e in output.tool_evidence if e["kind"] == "file_write"]) == 1
    finally:
        CURRENT_EXECUTION.reset(token)
        session.close()


def test_parallel_check_started_before_write_cannot_promote_task():
    ledger = TaskLedger(tasks=[Task("T1", "Edit", "x", "fast")])
    ledger.begin_task("T1")
    ledger.set_deliverable("T1", '{"summary":"done"}', trusted_evidence=[
        {"kind": "file_write", "reference": "a.py", "ok": True, "applied": True,
         "sequence": 1, "completed_at_ns": 200},
        {"kind": "test", "reference": "pytest", "ok": True, "sequence": 2,
         "started_at_ns": 100, "completed_at_ns": 300},
    ])
    assert not ledger.tasks[0].artifact.tested


def test_compaction_removes_only_exact_duplicates_and_keeps_full_state():
    first = ToolMessage(content="same data " * 100, name="lire_fichier", tool_call_id="first")
    unique = ToolMessage(content="unique data " * 100, name="lire_fichier", tool_call_id="unique")
    second = ToolMessage(content=first.content, name="lire_fichier", tool_call_id="second")
    state = {"messages": [first, unique, second]}
    compact = compact_duplicate_tool_results(state)["llm_input_messages"]
    assert "duplicate_of" in compact[0].content
    assert compact[1].content == unique.content
    assert compact[2].content == first.content
    assert state["messages"][0].content == first.content


def test_reviewer_can_read_files_and_its_cost_is_counted(workfile, tmp_path):
    _path, relative = workfile
    model = ScriptedModel(responses=[
        ai(tool_calls=[{"name": "lire_fichier", "args": {"chemin": relative}, "id": "review-read", "type": "tool_call"}]),
        ai("VERDICT: A CORRIGER\n- Le fichier contient value=1 ; ne pas annoncer VERDICT: CONFORME."),
    ])
    def factory(**kwargs):
        return create_react_agent(model, READ_ONLY_TOOLS, checkpointer=kwargs["checkpointer"])
    chief = ChefDeProjet()
    task = Task("T1", "Corriger", "Changer la valeur.", "fast", requires_review=True,
                acceptance_criteria=["La valeur vaut 2."], expected_output="Fichier corrigé.")
    chief.ledger = TaskLedger(tasks=[task])
    chief.ledger.begin_task("T1")
    chief.ledger.set_deliverable("T1", '{"summary":"Valeur changée."}', trusted_evidence=[
        {"kind": "file_write", "reference": relative, "ok": True, "applied": True, "sequence": 1}])
    with patch("core.multi_agent.orchestrator.get_workspace_reader_agent", side_effect=factory), \
         patch("tools.custom_tools.PROJECT_ROOT", tmp_path):
        verdicts = chief.review_flagged_tasks()
    assert verdicts == [("T1", "a corriger")]
    assert chief.ledger.metrics_summary()["coordination_model_calls"] == 2


def test_textual_result_continues_without_losing_whitespace(tmp_path):
    first = ai('{"summary":"Analyse","result":"première ')
    first.response_metadata = {"stopReason": "max_tokens"}
    model = ScriptedModel(responses=[first, ai('suite"}')])
    session = ExecutionSession("text-result", tmp_path)
    token = CURRENT_EXECUTION.set(session)
    try:
        graph = create_react_agent(model, [], checkpointer=session.checkpointer())
        output = invoke_agent_with_continuation(graph, [("user", "analyse")], session=session, complete_result=True)
        assert json.loads(output.text)["result"] == "première suite"
        assert model.position == 2
    finally:
        CURRENT_EXECUTION.reset(token)
        session.close()


def test_truncated_edit_report_costs_no_continuation_and_is_not_failure(workfile, tmp_path):
    _path, relative = workfile
    report = ai('{"summary":"Rapport coupé')
    report.response_metadata = {"stopReason": "max_tokens"}
    model = ScriptedModel(responses=[ai(tool_calls=[{"name": "remplacer_bloc_directement", "args": {
        "chemin": relative, "ancien_texte": "value = 1", "nouveau_texte": "value = 5"
    }, "id": "final-write", "type": "tool_call"}]), report])
    def factory(**kwargs):
        return create_react_agent(model, DIRECT_WRITE_TOOLS, checkpointer=kwargs["checkpointer"])
    with patch("core.multi_agent.workers.get_workspace_editor_agent", side_effect=factory), \
         patch("core.multi_agent.workers.PROJECT_ROOT", tmp_path):
        task = Task("T1", "Edit", "Modifier.", "fast", capabilities=["write"])
        ledger = TaskLedger(tasks=[task])
        ledger.begin_task("T1")
        result = run_single_task(task)
        apply_results(ledger, [result])
    assert result.ok
    assert model.position == 2
    assert result.continuations == 0
    assert ledger.tasks[0].status == "termine"
    assert ledger.tasks[0].artifact.state == "applied"
