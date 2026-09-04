"""
Persistance sur disque des conversations de l'assistant.

Objectifs :
- survivre à un redémarrage de l'application Streamlit ;
- conserver plusieurs conversations (création non destructive) ;
- borner l'espace disque (nombre de conversations + messages par conversation) ;
- ne persister que l'essentiel (rôle, contenu, libellé du modèle) ;
- stocker une mémoire compacte par conversation (résumé, tâches, décisions).

Un fichier JSON est créé par conversation dans PROJECT_ROOT/conversations.
L'écriture est atomique (fichier temporaire puis remplacement).

Ce module est volontairement indépendant de Streamlit et de LangChain
afin de rester simple à faire évoluer et à tester.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.custom_tools import PROJECT_ROOT


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

CONVERSATIONS_ROOT = (
    PROJECT_ROOT / "conversations"
).resolve()

CONVERSATIONS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

# Nombre maximal de conversations conservées sur disque.
# Les plus anciennes (updated_at) sont supprimées au-delà.
MAX_CONVERSATIONS = 30

# Nombre maximal de messages persistés par conversation.
# Seuls les derniers messages sont conservés au-delà.
MAX_MESSAGES_PER_CONV = 200

# Longueur maximale du titre déduit du premier message utilisateur.
MAX_TITLE_LENGTH = 60


# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------

class ConversationStoreError(RuntimeError):
    """Erreur de lecture ou d'écriture d'une conversation."""


# -------------------------------------------------------------------
# Fonctions internes
# -------------------------------------------------------------------

def _now_iso() -> str:
    """Retourne l'horodatage courant au format ISO 8601 (UTC)."""

    return datetime.now(timezone.utc).isoformat()


def _is_valid_id(conversation_id: str) -> bool:
    """Vérifie qu'un identifiant est un UUID hexadécimal simple."""

    if not isinstance(conversation_id, str):
        return False

    return bool(
        re.fullmatch(r"[0-9a-fA-F-]{16,64}", conversation_id)
    )


def _conversation_path(conversation_id: str) -> Path:
    """Retourne le chemin du fichier d'une conversation."""

    if not _is_valid_id(conversation_id):
        raise ConversationStoreError(
            "Identifiant de conversation invalide."
        )

    return CONVERSATIONS_ROOT / f"{conversation_id}.json"


def _empty_memory() -> dict[str, Any]:
    """Retourne une structure de mémoire vide."""

    return {
        "summary": "",
        "tasks": [],
        "decisions": [],
        "files": [],
    }


def _sanitize_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Ne conserve que les champs utiles et bornne le nombre de messages.

    Les métadonnées lourdes (usage, tool_call_count) ne sont pas
    persistées afin de limiter la taille du stockage.
    """

    sanitized: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role not in ("user", "assistant"):
            continue

        entry: dict[str, Any] = {
            "role": role,
            "content": str(message.get("content", "")),
        }

        model_label = message.get("model_label")

        if role == "assistant" and model_label:
            entry["model_label"] = str(model_label)

        sanitized.append(entry)

    if len(sanitized) > MAX_MESSAGES_PER_CONV:
        sanitized = sanitized[-MAX_MESSAGES_PER_CONV:]

    return sanitized


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """
    Écrit le contenu JSON de façon atomique.

    Un fichier temporaire est écrit puis renommé sur la cible,
    ce qui évite un fichier partiellement écrit en cas d'incident.
    """

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                payload,
                file_handle,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temporary_path, path)

    except OSError as error:
        raise ConversationStoreError(
            f"Écriture impossible : {error}"
        ) from error

    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _derive_title(messages: list[dict[str, Any]]) -> str:
    """
    Déduit un titre lisible à partir du premier message utilisateur.
    """

    for message in messages:
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()

            if content:
                single_line = " ".join(content.split())

                if len(single_line) > MAX_TITLE_LENGTH:
                    return (
                        single_line[:MAX_TITLE_LENGTH].rstrip()
                        + "…"
                    )

                return single_line

    return "Nouvelle conversation"


# -------------------------------------------------------------------
# API publique
# -------------------------------------------------------------------

def new_conversation_id() -> str:
    """Génère un identifiant de conversation unique."""

    return uuid.uuid4().hex


def list_conversations() -> list[dict[str, Any]]:
    """
    Retourne la liste des conversations, de la plus récente à la plus
    ancienne. Chaque entrée contient id, title, created_at, updated_at
    et le nombre de messages (métadonnées uniquement, sans le contenu).
    """

    summaries: list[dict[str, Any]] = []

    for path in CONVERSATIONS_ROOT.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)

        except (OSError, json.JSONDecodeError):
            continue

        summaries.append(
            {
                "id": data.get("id", path.stem),
                "title": data.get("title", "Conversation"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(
                    data.get("messages", [])
                ),
            }
        )

    summaries.sort(
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )

    return summaries


def load_conversation(
    conversation_id: str,
) -> dict[str, Any] | None:
    """
    Charge une conversation complète depuis le disque.

    Retourne None si le fichier est absent ou illisible.
    """

    path = _conversation_path(conversation_id)

    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

    except (OSError, json.JSONDecodeError):
        return None

    data.setdefault("messages", [])
    data.setdefault("memory", _empty_memory())

    return data


def save_conversation(
    conversation_id: str,
    messages: list[dict[str, Any]],
    memory: dict[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """
    Enregistre (ou met à jour) une conversation sur le disque.

    Ne persiste que les champs utiles des messages et applique
    les limites de taille. Applique aussi la limite globale du
    nombre de conversations conservées.
    """

    sanitized_messages = _sanitize_messages(messages)

    if not sanitized_messages:
        # Rien d'utile à conserver : on n'écrit pas de fichier vide.
        return {
            "id": conversation_id,
            "messages": [],
            "memory": memory or _empty_memory(),
        }

    existing = load_conversation(conversation_id)

    created_at = (
        existing.get("created_at")
        if existing
        else None
    ) or _now_iso()

    resolved_title = (
        title
        or (existing.get("title") if existing else None)
        or _derive_title(sanitized_messages)
    )

    payload = {
        "id": conversation_id,
        "title": resolved_title,
        "created_at": created_at,
        "updated_at": _now_iso(),
        "messages": sanitized_messages,
        "memory": memory or (
            existing.get("memory")
            if existing
            else _empty_memory()
        ),
    }

    _atomic_write(
        _conversation_path(conversation_id),
        payload,
    )

    enforce_conversation_limit()

    return payload


def delete_conversation(conversation_id: str) -> bool:
    """
    Supprime une conversation. Retourne True si un fichier a été retiré.
    """

    path = _conversation_path(conversation_id)

    if not path.exists():
        return False

    try:
        path.unlink()
        return True

    except OSError as error:
        raise ConversationStoreError(
            f"Suppression impossible : {error}"
        ) from error


def enforce_conversation_limit() -> int:
    """
    Supprime les conversations les plus anciennes au-delà de la limite.

    Retourne le nombre de conversations supprimées.
    """

    summaries = list_conversations()

    if len(summaries) <= MAX_CONVERSATIONS:
        return 0

    to_remove = summaries[MAX_CONVERSATIONS:]
    removed = 0

    for summary in to_remove:
        try:
            if delete_conversation(summary["id"]):
                removed += 1

        except ConversationStoreError:
            continue

    return removed
