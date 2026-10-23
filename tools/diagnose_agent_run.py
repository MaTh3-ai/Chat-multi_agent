"""Inspect local usage records without a model call or printing project contents.

Usage: python tools/diagnose_agent_run.py RUN_ID [--skip-checkpoints]
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3


METRICS = ("model_calls", "input_tokens", "output_tokens", "total_tokens",
           "cache_read_tokens", "cache_write_tokens", "unknown_usage_calls", "retry_count")


def metrics_row(label, metrics):
    return {"step": label, **{name: metrics.get(name) for name in METRICS}}


def inspect_messages(messages):
    """Report sizes and actual usage, never prompts, code, output or credentials."""
    calls, largest = [], []
    counts, characters, duplicates, reads = Counter(), Counter(), Counter(), Counter()
    seen = set()
    for message in messages:
        kind = getattr(message, "type", "")
        if kind == "ai":
            usage = getattr(message, "usage_metadata", None) or {}
            inputs = usage.get("input_token_details") or {}
            outputs = usage.get("output_token_details") or {}
            calls.append({"call": len(calls) + 1,
                          "input_tokens": usage.get("input_tokens"),
                          "output_tokens": usage.get("output_tokens"),
                          "cache_read_tokens": inputs.get("cache_read"),
                          "cache_write_tokens": inputs.get("cache_creation"),
                          "reasoning_tokens": outputs.get("reasoning")})
        if kind != "tool":
            continue
        name = getattr(message, "name", None) or "unknown"
        content = message.content
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        counts[name] += 1
        characters[name] += len(text)
        key = (name, hashlib.sha256(text.encode()).hexdigest())
        duplicates[name] += int(key in seen)
        seen.add(key)
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            payload = None
        path = payload.get("path") if isinstance(payload, dict) else None
        largest.append({"tool": name, "path": path, "characters": len(text)})
        if name in {"lire_fichier", "lire_extraits_fichiers"} and isinstance(payload, dict):
            for item in [payload, *(payload.get("files") or [])]:
                if isinstance(item, dict) and item.get("ok") and item.get("path"):
                    reads[str(item["path"])] += 1
    return {"saved_model_calls": calls,
            "tools": [{"tool": name, "calls": counts[name], "stored_result_characters": characters[name],
                       "exact_duplicate_results": duplicates[name]} for name in counts],
            "file_reads": dict(reads),
            "largest_tool_results": sorted(largest, key=lambda item: -item["characters"])[:8]}


def inspect_checkpoints(path, tasks):
    # The saver understands its versioned serialization; the connection is read-only.
    from langgraph.checkpoint.sqlite import SqliteSaver
    result = []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        saver = SqliteSaver(connection)
        threads = [row[0] for row in connection.execute("SELECT DISTINCT thread_id FROM checkpoints")]
        for task in tasks:
            identity = task.get("execution_id")
            if not identity:
                continue
            for thread in threads:
                if not thread.startswith(identity + ":"):
                    continue
                snapshot = saver.get_tuple({"configurable": {"thread_id": thread, "checkpoint_ns": ""}})
                if snapshot:
                    messages = snapshot.checkpoint.get("channel_values", {}).get("messages", [])
                    result.append({"task": task["id"], "thread": thread, **inspect_messages(messages)})
    return result


def diagnose(path, checkpoint_path=None):
    data = json.loads(path.read_text(encoding="utf-8"))
    ledger = data.get("ledger", data)
    tasks = ledger.get("tasks", [])
    steps = [metrics_row(task.get("id", "?") + " / " + str(task.get("title", ""))[:120],
                         task.get("metrics") or {}) for task in tasks]
    steps.append(metrics_row("coordination", ledger.get("coordination_metrics") or {}))
    report = {"run_id": ledger.get("run_id"), "state": ledger.get("state"), "steps": steps,
              "totals": {name: sum(row[name] or 0 for row in steps) for name in METRICS},
              "notes": ["Tokens cumulés des appels, pas longueur du code ni du résultat final.",
                        "Le ratio de répétition de l'UI historique mesure seulement le rapport final.",
                        "Les checkpoints contiennent un historique de session : ne pas ajouter ses usages aux totaux du ledger.",
                        "reasoning_tokens=null signifie que le fournisseur ne l'a pas renseigné."]}
    if checkpoint_path and checkpoint_path.is_file():
        try:
            report["sessions"] = inspect_checkpoints(checkpoint_path, tasks)
        except (ImportError, sqlite3.Error, ValueError, TypeError) as error:
            report["checkpoint_diagnostic_unavailable"] = type(error).__name__ + ": " + str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Run ID ou chemin du fichier de sauvegarde JSON")
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()
    root = Path(os.getenv("APP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    supplied = Path(args.run)
    path = supplied if supplied.is_file() else root / "agent_runs" / (args.run + ".json")
    if not path.is_file():
        parser.error(f"Sauvegarde introuvable : {path}")
    checkpoint = None if args.skip_checkpoints else root / ".agent_state" / "execution.sqlite3"
    try:
        print(json.dumps(diagnose(path, checkpoint), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
