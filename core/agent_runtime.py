"""Structured execution records; no model is needed to maintain this state.

Shared by tools and workers (outside multi_agent to avoid import cycles).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from functools import wraps
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class ExecutionCancelled(RuntimeError):
    """Cooperative pause before the next paid inference."""


class ExecutionBudgetExceeded(RuntimeError):
    """A new inference was refused at a model boundary; tools already ran."""


@dataclass
class RunBudget:
    # Zero means unconfigured, not a silently imposed task budget.
    max_calls: int = 0
    max_tokens: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    unknown_usage_calls: int = 0
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def from_env(cls) -> "RunBudget":
        return cls(
            max_calls=max(0, int(os.getenv("AGENT_RUN_MAX_CALLS", "0"))),
            max_tokens=max(0, int(os.getenv("AGENT_RUN_MAX_TOKENS", "0"))),
        )

    def begin_call(self) -> None:
        with self._lock:
            if self.max_calls and self.model_calls >= self.max_calls:
                raise ExecutionBudgetExceeded("Budget d'appels atteint ; travail conservé.")
            if self.max_tokens and self.input_tokens + self.output_tokens >= self.max_tokens:
                raise ExecutionBudgetExceeded("Budget de tokens atteint ; travail conservé.")
            self.model_calls += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {name: getattr(self, name) for name in (
                "model_calls", "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "unknown_usage_calls",
            )}


class UsageRecorder(BaseCallbackHandler):
    """Actual successful usage plus attempted calls, including failed attempts."""
    raise_error = True

    def __init__(self, budget: RunBudget, parent: RunBudget | None = None, cancel_event=None):
        self.budget = budget
        self.parent = parent
        self.cancel_event = cancel_event

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ExecutionCancelled("Exécution suspendue ; travail conservé.")
        if self.parent is not None:
            self.parent.begin_call()
        self.budget.begin_call()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        message = None
        if getattr(response, "generations", None):
            message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None)
        for budget in [self.budget] + ([self.parent] if self.parent is not None else []):
            with budget._lock:
                if not isinstance(usage, dict) or not usage:
                    budget.unknown_usage_calls += 1
                    continue
                budget.input_tokens += int(usage.get("input_tokens") or 0)
                budget.output_tokens += int(usage.get("output_tokens") or 0)
                details = usage.get("input_token_details") or {}
                budget.cache_read_tokens += int(details.get("cache_read") or 0)
                budget.cache_write_tokens += int(details.get("cache_creation") or (
                    (details.get("ephemeral_5m_input_tokens") or 0) +
                    (details.get("ephemeral_1h_input_tokens") or 0)))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        # A timeout is not proof that the provider did not bill the call.
        if isinstance(error, (ExecutionBudgetExceeded, ExecutionCancelled)):
            return
        for budget in [self.budget] + ([self.parent] if self.parent is not None else []):
            with budget._lock:
                budget.unknown_usage_calls += 1


@dataclass
class ExecutionSession:
    key: str
    root: Path
    budget: RunBudget = field(default_factory=RunBudget)
    run_budget: RunBudget | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    cancel_event: Any = None
    context_update: str = ""
    check_paths: set[str] = field(default_factory=set)
    agent: Any = None
    messages: list[Any] = field(default_factory=list)
    read_versions: dict[str, str] = field(default_factory=dict)
    writes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _connection: Any = None
    _checkpoint_connection: Any = None
    _lock: Any = field(default_factory=threading.RLock, repr=False)

    def connect(self):
        if self._connection is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.root / "execution.sqlite3", check_same_thread=False)
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS applied_tools "
                "(session TEXT, operation TEXT, result TEXT, PRIMARY KEY(session, operation))"
            )
        return self._connection

    def checkpointer(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        self.connect()
        if self._checkpoint_connection is None:
            # Graph checkpoints can be written by a background thread while
            # a tool journals a write. Separate connections avoid sharing an
            # open transaction; SQLite serializes the database writes.
            self._checkpoint_connection = sqlite3.connect(self.root / "execution.sqlite3",
                                                         check_same_thread=False, timeout=30)
        return SqliteSaver(self._checkpoint_connection)

    def observe(self, path: Path) -> None:
        if path.is_file():
            with self._lock:
                self.read_versions[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def operation_key(self, name: str, arguments: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps([name, arguments], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def replay(self, operation: str, path: Path) -> dict[str, Any] | None:
        with self._lock:
            row = self.connect().execute(
                "SELECT result FROM applied_tools WHERE session=? AND operation=?", (self.key, operation)
            ).fetchone()
        if not row:
            return None
        result = json.loads(row[0])
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if result.get("content_hash") == current:
            return {**result, "replayed": True, "changed": False}
        return None

    def record_write(self, operation: str, result: dict[str, Any]) -> None:
        with self._lock:
            self.connect().execute("INSERT OR REPLACE INTO applied_tools VALUES (?, ?, ?)",
                                   (self.key, operation, json.dumps(result)))
            self.connect().commit()
            self.writes[str(result["path"])] = result

    def close(self) -> None:
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None


CURRENT_EXECUTION: ContextVar[ExecutionSession | None] = ContextVar("agent_execution", default=None)
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def path_lock(path: Path):
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(str(path.resolve()), threading.RLock())


def command_category(command: list[str]) -> str:
    """Classify execution, not the model's assertion that it was a test."""
    executable = Path(command[0]).name.casefold().removesuffix(".exe") if command else ""
    args = command[1:]
    if any(value in {"--help", "-h", "--version", "--collect-only", "--co"} for value in args):
        return "command"
    if executable in {"python", "python3", "py"} and len(args) >= 2 and args[0] == "-m":
        executable, args = args[1], args[2:]
    if executable in {"pytest", "unittest"}:
        return "test"
    if executable in {"npm", "pnpm", "yarn"}:
        script = args[1] if args[:1] == ["run"] and len(args) > 1 else (args[0] if args else "")
        if script == "test" or script.startswith("test:"):
            return "test"
        if script in {"lint", "typecheck", "build"}:
            return script
    if executable == "ruff" and args[:1] == ["check"]:
        return "lint"
    if executable in {"py_compile", "compileall", "tsc"}:
        return "syntax"
    return "command"


# Commands may edit any project file. Serialize mutation windows, while model
# inference and read-only work remain parallel. RLock permits batch wrappers.
WORKSPACE_MUTATION_LOCK = threading.RLock()


def workspace_mutation(function):
    @wraps(function)
    def locked(*args, **kwargs):
        with WORKSPACE_MUTATION_LOCK:
            return function(*args, **kwargs)
    return locked


def snapshot_files(root: Path, ignored: set[str]) -> dict[str, str]:
    """Content snapshot; prune ignored trees before visiting their children."""
    result = {}
    for directory, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name not in ignored
                          and not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                result[path.relative_to(root).as_posix()] = digest.hexdigest()
            except OSError:
                # Do not certify a partially observed workspace.
                raise RuntimeError(f"Impossible d'observer {path}")
    return result
