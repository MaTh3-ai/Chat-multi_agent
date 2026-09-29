"""Journal partagé des commandes exécutées par les agents.

Les workers s'exécutent dans des threads : le journal est protégé par un
verrou et borné, afin de rester consultable sans faire croître la mémoire.
Il sert de trace unique pour le mode Chat comme pour le mode Multi-agent.
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

MAX_JOURNAL_ENTRIES = 200
MAX_STREAM_CHARACTERS = 4_000

_JOURNAL: deque[dict[str, Any]] = deque(maxlen=MAX_JOURNAL_ENTRIES)
_LOCK = threading.Lock()
_COUNTER = 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_command(
    *,
    command: list[str] | str,
    cwd: str = "",
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
    error: str = "",
    duration_seconds: float | None = None,
    source: str = "agent",
) -> dict[str, Any]:
    """Enregistre une exécution (ou un refus) et retourne l'entrée créée."""
    global _COUNTER
    if isinstance(command, str):
        display = command
        parts = [command]
    else:
        parts = [str(item) for item in command]
        display = " ".join(parts)
    entry = {
        "command": parts,
        "display": display,
        "cwd": cwd,
        "returncode": returncode,
        "stdout": (stdout or "")[-MAX_STREAM_CHARACTERS:],
        "stderr": (stderr or "")[-MAX_STREAM_CHARACTERS:],
        "error": error,
        "duration_seconds": duration_seconds,
        "source": source,
        "timestamp": _utc_now_iso(),
        # Un refus n'a pas de code retour : il ne doit pas passer pour un succès.
        "ok": bool(not error and returncode == 0),
    }
    with _LOCK:
        _COUNTER += 1
        entry["sequence"] = _COUNTER
        _JOURNAL.append(entry)
    return entry


def get_command_journal(limit: int | None = None) -> list[dict[str, Any]]:
    """Retourne les entrées de la plus récente à la plus ancienne."""
    with _LOCK:
        entries = list(_JOURNAL)
    entries.reverse()
    if limit is not None and limit >= 0:
        return entries[:limit]
    return entries


def command_journal_size() -> int:
    with _LOCK:
        return len(_JOURNAL)


def clear_command_journal() -> int:
    with _LOCK:
        removed = len(_JOURNAL)
        _JOURNAL.clear()
    return removed


def render_command_journal(
    *,
    key_prefix: str,
    limit: int = 25,
    expanded: bool = False,
) -> None:
    """Affiche le journal dans Streamlit (import local : module réutilisable hors UI)."""
    import streamlit as st

    entries = get_command_journal(limit=limit)
    total = command_journal_size()
    if not entries:
        st.caption("Aucune commande executee pour le moment.")
        return

    failures = sum(1 for entry in entries if not entry["ok"])
    title = f"Journal des commandes ({total})"
    if failures:
        title += f" — {failures} en echec"
    with st.expander(title, expanded=expanded):
        if st.button("Vider le journal", key=f"{key_prefix}_clear_commands"):
            clear_command_journal()
            st.rerun()
        for entry in entries:
            icon = "✅" if entry["ok"] else "❌"
            header = f"{icon} `{entry['display']}`"
            if entry["cwd"]:
                header += f" — dossier : `{entry['cwd']}`"
            st.markdown(header)
            details = [entry["timestamp"]]
            if entry["returncode"] is not None:
                details.append(f"code retour : {entry['returncode']}")
            if entry["duration_seconds"] is not None:
                details.append(f"{entry['duration_seconds']:.1f}s")
            st.caption(" • ".join(details))
            if entry["error"]:
                st.error(entry["error"])
            if entry["stdout"]:
                st.code(entry["stdout"], language="text")
            if entry["stderr"]:
                st.markdown("_stderr_")
                st.code(entry["stderr"], language="text")
            st.divider()
