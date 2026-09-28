"""Coordination compacte des workers : revisions, preuves et contexte cible."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.custom_tools import WORKSPACE_ROOT

ARTIFACT_STATES = (
    "observed",
    "proposed",
    "approved",
    "applied",
    "tested",
    "verified",
)
STATE_RANK = {state: index for index, state in enumerate(ARTIFACT_STATES)}
DEFAULT_CONTEXT_TARGET_CHARACTERS = 12_000
MAX_DISCOVERY_CONTEXT_CHARACTERS = 6_000
MAX_DEPENDENCY_CONTEXT_CHARACTERS = 2_000
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class Evidence:
    kind: str
    reference: str
    detail: str = ""
    content_hash: str | None = None


@dataclass
class Artifact:
    id: str
    task_id: str
    kind: str
    state: str
    summary: str
    references: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    input_revision: str = ""
    output_revision: str = ""
    tested: bool = False
    claims: list[dict[str, str]] = field(default_factory=list)


@dataclass
class WorkerEnvelope:
    state: str
    summary: str
    references: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    claims: list[dict[str, str]] = field(default_factory=list)


def workspace_revision(root: Path = WORKSPACE_ROOT) -> str:
    """Calcule une revision stable et peu couteuse du workspace."""
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            stat = path.stat()
            relative = path.relative_to(root)
        except (OSError, ValueError):
            continue
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()


def file_content_hash(relative_path: str, root: Path = WORKSPACE_ROOT) -> str | None:
    try:
        path = (root / relative_path).resolve()
        path.relative_to(root.resolve())
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None


def parse_worker_envelope(text: str) -> WorkerEnvelope:
    """Parse la sortie JSON courte, avec repli factuel non bloquant."""
    candidate = text.strip()
    fenced = _FENCE_RE.fullmatch(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        raw = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return WorkerEnvelope(
            state="observed",
            summary=(first_line or "Livrable textuel produit.")[:500],
            blockers=["Sortie non structurée ; preuves non indexées automatiquement."],
        )
    if not isinstance(raw, dict):
        return WorkerEnvelope(state="observed", summary="Livrable JSON non objet.")
    state = str(raw.get("state", "observed")).strip().casefold()
    if state not in STATE_RANK:
        state = "observed"
    evidence: list[Evidence] = []
    for item in raw.get("evidence", []) if isinstance(raw.get("evidence"), list) else []:
        if not isinstance(item, dict):
            continue
        reference = str(item.get("reference") or item.get("path") or "").strip()
        if not reference:
            continue
        evidence.append(Evidence(
            kind=str(item.get("kind") or item.get("type") or "reference"),
            reference=reference,
            detail=str(item.get("detail") or ""),
            content_hash=str(item.get("hash") or "") or file_content_hash(reference),
        ))
    return WorkerEnvelope(
        state=state,
        summary=str(raw.get("summary") or "Résultat produit.").strip()[:1000],
        references=[str(item) for item in raw.get("references", []) if str(item).strip()]
        if isinstance(raw.get("references"), list) else [],
        evidence=evidence,
        proposal_ids=[str(item) for item in raw.get("proposal_ids", []) if str(item).strip()]
        if isinstance(raw.get("proposal_ids"), list) else [],
        tests=[str(item) for item in raw.get("tests", []) if str(item).strip()]
        if isinstance(raw.get("tests"), list) else [],
        blockers=[str(item) for item in raw.get("blockers", []) if str(item).strip()]
        if isinstance(raw.get("blockers"), list) else [],
        claims=[
            {
                "subject": str(item.get("subject", "")).strip(),
                "predicate": str(item.get("predicate", "")).strip(),
                "value": str(item.get("value", "")).strip(),
            }
            for item in raw.get("claims", [])
            if isinstance(item, dict)
            and str(item.get("subject", "")).strip()
            and str(item.get("predicate", "")).strip()
        ] if isinstance(raw.get("claims"), list) else [],
    )


def compact_dependency(task: Any) -> str:
    artifact = getattr(task, "artifact", None)
    if artifact:
        references = ", ".join(artifact.references[:8]) or "aucune"
        return (
            f"{task.id} | état={artifact.state} | {artifact.summary}\n"
            f"références={references} | révision={artifact.output_revision[:12]}"
        )[:MAX_DEPENDENCY_CONTEXT_CHARACTERS]
    deliverable = str(getattr(task, "deliverable", "") or "")
    return f"{task.id} | {deliverable[:MAX_DEPENDENCY_CONTEXT_CHARACTERS]}"


def compile_context(ledger: Any, task: Any, discovery_context: str = "") -> str:
    """Construit un contexte cible sans recopier les rapports complets."""
    parts: list[str] = []
    if discovery_context:
        parts.append(
            "DÉCOUVERTE PARTIELLE NON AUTORITATIVE\n"
            + discovery_context[:MAX_DISCOVERY_CONTEXT_CHARACTERS]
        )
    dependencies = []
    for dependency_id in getattr(task, "dependencies", []):
        dependency = ledger.get_task(dependency_id)
        if dependency:
            dependencies.append(compact_dependency(dependency))
    if dependencies:
        parts.append("ARTEFACTS DÉPENDANTS\n" + "\n".join(dependencies))
    contradictions = [
        item for item in getattr(ledger, "contradictions", [])
        if not item.get("resolved", False)
        and (not item.get("task_ids") or task.id in item.get("task_ids", []))
    ]
    if contradictions:
        lines = [
            f"{item.get('id')}: {item.get('message')}"
            for item in contradictions[:5]
        ]
        parts.append("CONTRADICTIONS OUVERTES\n" + "\n".join(lines))
    return "\n\n".join(parts)[:DEFAULT_CONTEXT_TARGET_CHARACTERS]


def repetition_ratio(text: str) -> float:
    lines = [line.strip().casefold() for line in text.splitlines() if len(line.strip()) >= 20]
    if not lines:
        return 0.0
    return max(0.0, min(1.0, (len(lines) - len(set(lines))) / len(lines)))
