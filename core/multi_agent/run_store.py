"""Persistance JSON atomique et bornee des chantiers multi-agent.

Ce module comble l'absence d'historique du mode agent. Il reutilise le meme
patron que core.conversation_store (ecriture atomique via fichier temporaire +
os.replace, verrou reentrant, bornes de taille) et s'appuie sur la
(de)serialisation deja fournie par TaskLedger.to_dict() / from_dict().

Il ne modifie aucun comportement existant : c'est un ajout autonome que l'UI
peut appeler pour sauvegarder et restaurer un chantier.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.multi_agent.ledger import TaskLedger
from tools.custom_tools import PROJECT_ROOT

RUNS_ROOT = (PROJECT_ROOT / "agent_runs").resolve()
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
MAX_RUNS = 30
MAX_TITLE_LENGTH = 80
SCHEMA_VERSION = 1
_STORE_LOCK = threading.RLock()


class RunStoreError(RuntimeError):
    """Erreur de lecture, validation ou ecriture d'un chantier."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_valid_id(run_id: str) -> bool:
    if not isinstance(run_id, str):
        return False
    try:
        uuid.UUID(run_id)
        return True
    except (ValueError, AttributeError):
        return bool(re.fullmatch(r"[0-9a-fA-F]{32}", run_id))


def _run_path(run_id: str) -> Path:
    if not _is_valid_id(run_id):
        raise RunStoreError("Identifiant de chantier invalide.")
    return RUNS_ROOT / f"{run_id}.json"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as error:
        raise RunStoreError(f"Ecriture impossible : {error}") from error
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    """Lit un fichier JSON et retourne None si corrompu ou absent.
    
    Gestion explicite de corruption : retourne None au lieu de lever une exception,
    permettant à load_run() de signaler clairement le problème.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError:
        # Fichier absent ou inaccessible
        return None
    except json.JSONDecodeError as error:
        # JSON corrompu : on retourne None et load_run() lèvera RunStoreError
        return None
    return value if isinstance(value, dict) else None


def _validate_schema(data: dict[str, Any]) -> list[str]:
    """Valide la structure du fichier de chantier.
    
    Retourne une liste d'erreurs de validation. Liste vide = valide.
    """
    errors = []
    
    if "schema_version" not in data:
        errors.append("Champ 'schema_version' manquant")
    elif not isinstance(data["schema_version"], int):
        errors.append("'schema_version' doit être un entier")
    
    if "id" not in data:
        errors.append("Champ 'id' manquant")
    elif not _is_valid_id(data["id"]):
        errors.append("'id' invalide (doit être un UUID)")
    
    if "ledger" not in data:
        errors.append("Champ 'ledger' manquant")
    elif not isinstance(data["ledger"], dict):
        errors.append("'ledger' doit être un dictionnaire")
    else:
        # Validation minimale du ledger
        ledger = data["ledger"]
        required_ledger_fields = ["state", "tasks", "run_id"]
        for field in required_ledger_fields:
            if field not in ledger:
                errors.append(f"Champ 'ledger.{field}' manquant")
    
    return errors


def _derive_title(ledger: TaskLedger) -> str:
    text = " ".join(str(ledger.objective or "").split())
    if text:
        return text[:MAX_TITLE_LENGTH].rstrip() + ("..." if len(text) > MAX_TITLE_LENGTH else "")
    return "Chantier sans objectif"


def _enforce_quota() -> None:
    """Conserve au plus MAX_RUNS chantiers, supprime les plus anciens."""
    with _STORE_LOCK:
        paths = list(RUNS_ROOT.glob("*.json"))
    if len(paths) <= MAX_RUNS:
        return
    dated = []
    for path in paths:
        data = _read_json(path)
        dated.append((data.get("updated_at", "") if data else "", path))
    dated.sort(key=lambda item: item[0])
    for _, path in dated[: len(dated) - MAX_RUNS]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def new_run_id() -> str:
    return uuid.uuid4().hex


def list_runs() -> list[dict[str, Any]]:
    """Liste tous les chantiers sauvegardés, triés par date de mise à jour.
    
    Ignore silencieusement les fichiers corrompus pour ne pas bloquer l'UI.
    """
    summaries: list[dict[str, Any]] = []
    with _STORE_LOCK:
        paths = list(RUNS_ROOT.glob("*.json"))
    for path in paths:
        data = _read_json(path)
        if not data:
            continue
        summaries.append(
            {
                "id": data.get("id", path.stem),
                "title": data.get("title", "Chantier"),
                "state": data.get("state", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "task_count": data.get("task_count", 0),
            }
        )
    return sorted(summaries, key=lambda item: item.get("updated_at", ""), reverse=True)


def save_run(ledger: TaskLedger, title: str | None = None) -> dict[str, Any]:
    """Sauvegarde un chantier a partir de son TaskLedger.

    L'identifiant de fichier est le run_id du ledger. Retourne le resume ecrit.
    Applique automatiquement le quota après écriture.
    """
    run_id = ledger.run_id if _is_valid_id(ledger.run_id) else new_run_id()
    path = _run_path(run_id)
    with _STORE_LOCK:
        existing = _read_json(path)
        now = _now_iso()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "id": run_id,
            "title": (title or (existing or {}).get("title") or _derive_title(ledger))[:MAX_TITLE_LENGTH],
            "state": ledger.state,
            "task_count": len(ledger.tasks),
            "created_at": (existing or {}).get("created_at", now),
            "updated_at": now,
            "ledger": ledger.to_dict(),
        }
        _atomic_write(path, payload)
    _enforce_quota()
    return {
        "id": run_id,
        "title": payload["title"],
        "state": payload["state"],
        "task_count": payload["task_count"],
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
    }


def load_run(run_id: str) -> TaskLedger | None:
    """Recharge un chantier et reconstruit son TaskLedger, ou None si absent.
    
    Lève RunStoreError avec message explicite si le fichier est corrompu ou
    incompatible avec le schéma actuel. Ne retourne None que si le fichier
    est réellement absent.
    """
    path = _run_path(run_id)
    if not path.exists():
        return None
    
    data = _read_json(path)
    if data is None:
        raise RunStoreError(
            f"Fichier de chantier {run_id} corrompu : JSON invalide ou illisible. "
            "Supprime le fichier manuellement ou utilise delete_run()."
        )
    
    # Validation du schéma
    validation_errors = _validate_schema(data)
    if validation_errors:
        raise RunStoreError(
            f"Fichier de chantier {run_id} invalide : " + ", ".join(validation_errors)
        )
    
    # Reconstruction du TaskLedger
    if not isinstance(data.get("ledger"), dict):
        raise RunStoreError(
            f"Fichier de chantier {run_id} : champ 'ledger' absent ou invalide."
        )
    
    try:
        return TaskLedger.from_dict(data["ledger"])
    except (TypeError, ValueError, KeyError) as error:
        raise RunStoreError(
            f"Chantier {run_id} incompatible avec le schéma actuel : {error}. "
            "Le fichier provient peut-être d'une version antérieure."
        ) from error


def delete_run(run_id: str) -> bool:
    """Supprime un fichier de chantier.
    
    Retourne True si supprimé, False en cas d'erreur (pas d'exception).
    """
    path = _run_path(run_id)
    with _STORE_LOCK:
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            return False
