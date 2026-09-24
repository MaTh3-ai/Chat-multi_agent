"""Persistance JSON atomique et bornée des conversations."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.memory_tools import normalize_memory
from tools.custom_tools import PROJECT_ROOT

CONVERSATIONS_ROOT = (PROJECT_ROOT / "conversations").resolve()
CONVERSATIONS_ROOT.mkdir(parents=True, exist_ok=True)
MAX_CONVERSATIONS = 30
MAX_MESSAGES_PER_CONV = 200
MAX_MESSAGE_CHARACTERS = 100_000
MAX_TITLE_LENGTH = 80
SCHEMA_VERSION = 2
_STORE_LOCK = threading.RLock()


class ConversationStoreError(RuntimeError):
    """Erreur de lecture, validation ou écriture d'une conversation."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_valid_id(conversation_id: str) -> bool:
    if not isinstance(conversation_id, str):
        return False
    try:
        uuid.UUID(conversation_id)
        return True
    except (ValueError, AttributeError):
        return bool(re.fullmatch(r"[0-9a-fA-F]{32}", conversation_id))


def _conversation_path(conversation_id: str) -> Path:
    if not _is_valid_id(conversation_id):
        raise ConversationStoreError("Identifiant de conversation invalide.")
    return CONVERSATIONS_ROOT / f"{conversation_id}.json"


def _sanitize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content", ""))
        if len(content) > MAX_MESSAGE_CHARACTERS:
            content = content[:MAX_MESSAGE_CHARACTERS].rstrip() + "…"
        entry: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and message.get("model_label"):
            entry["model_label"] = str(message["model_label"])[:200]
        attachments = message.get("attachments")
        if role == "user" and isinstance(attachments, list):
            safe_attachments = []
            for item in attachments[:20]:
                if not isinstance(item, dict):
                    continue
                safe_attachments.append(
                    {
                        "kind": str(item.get("kind", ""))[:30],
                        "name": Path(str(item.get("name", "document"))).name[:255],
                        "format": str(item.get("format", ""))[:20],
                        "size": int(item.get("size", 0) or 0),
                    }
                )
            if safe_attachments:
                entry["attachments"] = safe_attachments
        sanitized.append(entry)
    return sanitized[-MAX_MESSAGES_PER_CONV:]


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as error:
        raise ConversationStoreError(f"Écriture impossible : {error}") from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _derive_title(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(str(message.get("content", "")).split())
            if text:
                return text[:MAX_TITLE_LENGTH].rstrip() + ("…" if len(text) > MAX_TITLE_LENGTH else "")
    return "Nouvelle conversation"


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def list_conversations() -> list[dict[str, Any]]:
    summaries = []
    with _STORE_LOCK:
        paths = list(CONVERSATIONS_ROOT.glob("*.json"))
    for path in paths:
        data = _read_json(path)
        if not data:
            continue
        summaries.append(
            {
                "id": data.get("id", path.stem),
                "title": data.get("title", "Conversation"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            }
        )
    return sorted(summaries, key=lambda item: item.get("updated_at", ""), reverse=True)


def load_conversation(conversation_id: str) -> dict[str, Any] | None:
    path = _conversation_path(conversation_id)
    if not path.exists():
        return None
    data = _read_json(path)
    if not data:
        return None
    data["messages"] = _sanitize_messages(data.get("messages", []))
    data["memory"] = normalize_memory(data.get("memory"))
    data.setdefault("schema_version", 1)
    return data


def save_conversation(
    conversation_id: str,
    messages: list[dict[str, Any]],
    memory: dict[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    sanitized = _sanitize_messages(messages)
    normalized_memory = normalize_memory(memory)
    if not sanitized:
        return {"id": conversation_id, "messages": [], "memory": normalized_memory}
    with _STORE_LOCK:
        existing = load_conversation(conversation_id)
        now = _now_iso()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "id": conversation_id,
            "title": (title or (existing or {}).get("title") or _derive_title(sanitized))[:MAX_TITLE_LENGTH],
            "created_at": (existing or {}).get("created_at") or now,
            "updated_at": now,
            "messages": sanitized,
            "memory": normalized_memory,
        }
        _atomic_write(_conversation_path(conversation_id), payload)
        enforce_conversation_limit()
        return payload


def delete_conversation(conversation_id: str) -> bool:
    path = _conversation_path(conversation_id)
    with _STORE_LOCK:
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as error:
            raise ConversationStoreError(f"Suppression impossible : {error}") from error


def enforce_conversation_limit() -> int:
    summaries = list_conversations()
    removed = 0
    for summary in summaries[MAX_CONVERSATIONS:]:
        try:
            removed += int(delete_conversation(summary["id"]))
        except ConversationStoreError:
            continue
    return removed
