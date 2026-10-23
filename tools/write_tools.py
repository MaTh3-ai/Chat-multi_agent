"""
Outils de proposition et d'application de modifications de fichiers.

Fonctionnalités :
- création ou remplacement complet d'un fichier texte ;
- remplacement ciblé d'un bloc de texte ;
- insertion avant ou après une ancre unique ;
- suppression ciblée d'un bloc de texte ;
- aperçu des différences avant application ;
- approbation obligatoire dans Streamlit ;
- sauvegarde avant modification ;
- écriture atomique ;
- détection des modifications concurrentes ;
- validation syntaxique de certains formats.

Les outils exposés au modèle créent uniquement des propositions.
Seule l'interface Streamlit peut approuver ou refuser celles-ci.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import threading
import tomllib
import uuid
import time
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from core.agent_runtime import CURRENT_EXECUTION, path_lock

from tools.custom_tools import (
    ALLOWED_TEXT_EXTENSIONS,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
    resolve_workspace_path,
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

BACKUP_ROOT = (
    PROJECT_ROOT / "backups"
).resolve()

BACKUP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_WRITE_CHARACTERS = 500_000
MAX_PATCH_ARGUMENT_CHARACTERS = 100_000
MAX_DIFF_CHARACTERS = 30_000

# Refus par défaut lorsqu'un remplacement complet supprime
# plus de 90 % du contenu d'un fichier existant.
MAX_AUTOMATIC_REDUCTION_RATIO = 0.90

_PENDING_PROPOSALS: dict[
    str,
    dict[str, Any],
] = {}
_RESOLVED_PROPOSAL_STATUSES: dict[str, str] = {}

_PROPOSAL_LOCK = threading.RLock()

PROPOSAL_ROOT = (PROJECT_ROOT / "proposals").resolve()
PROPOSAL_STORE_PATH = PROPOSAL_ROOT / "proposals.json"


# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------

class FileModificationError(RuntimeError):
    """Erreur pendant la préparation ou l'application d'une modification."""


class ProposalNotFoundError(FileModificationError):
    """La proposition demandée n'existe pas ou n'est plus disponible."""


class ConcurrentModificationError(FileModificationError):
    """Le fichier a changé depuis la création de la proposition."""


def _save_proposals_unlocked() -> None:
    """Persiste atomiquement les propositions. Verrou deja acquis."""
    PROPOSAL_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_path = PROPOSAL_STORE_PATH.with_name(
        f".{PROPOSAL_STORE_PATH.name}.{uuid.uuid4().hex}.tmp"
    )
    payload = {
        "schema_version": 2,
        "proposals": list(_PENDING_PROPOSALS.values()),
        "resolved_statuses": _RESOLVED_PROPOSAL_STATUSES,
    }
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(PROPOSAL_STORE_PATH)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise FileModificationError(
            f"Impossible de persister les propositions : {error}"
        ) from error


def _load_proposals_from_disk() -> None:
    """Recharge les propositions persistees sans bloquer le demarrage."""
    if not PROPOSAL_STORE_PATH.exists():
        return
    try:
        payload = json.loads(PROPOSAL_STORE_PATH.read_text(encoding="utf-8"))
        proposals = payload.get("proposals", [])
        if not isinstance(proposals, list):
            return
        with _PROPOSAL_LOCK:
            resolved = payload.get("resolved_statuses", {})
            if isinstance(resolved, dict):
                _RESOLVED_PROPOSAL_STATUSES.update(
                    {str(key): str(value) for key, value in resolved.items()}
                )
            for proposal in proposals:
                if isinstance(proposal, dict) and proposal.get("proposal_id"):
                    _PENDING_PROPOSALS[str(proposal["proposal_id"])] = proposal
    except (OSError, json.JSONDecodeError, TypeError):
        return


_load_proposals_from_disk()


# -------------------------------------------------------------------
# Fonctions internes de lecture et validation
# -------------------------------------------------------------------

def _validate_writable_file(
    path: Path,
) -> None:
    """
    Vérifie que le fichier possède un format texte autorisé.
    """

    extension = path.suffix.lower()

    if extension not in ALLOWED_TEXT_EXTENSIONS:
        allowed_extensions = ", ".join(
            sorted(ALLOWED_TEXT_EXTENSIONS)
        )

        raise FileModificationError(
            f"Format non autorisé pour l'écriture : "
            f"{extension or 'sans extension'}. "
            f"Formats autorisés : {allowed_extensions}"
        )


def _read_existing_content(
    path: Path,
) -> str:
    """
    Lit le contenu actuel du fichier.

    Retourne une chaîne vide lorsque le fichier n'existe pas.
    """

    if not path.exists():
        return ""

    if not path.is_file():
        raise FileModificationError(
            "Le chemin demandé existe, mais ce n'est pas un fichier."
        )

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except OSError as error:
        raise FileModificationError(
            f"Impossible de lire le fichier actuel : {error}"
        ) from error


def _content_hash(
    content: str,
) -> str:
    """
    Calcule une empreinte stable du contenu d'un fichier.
    """

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def _create_diff(
    relative_path: str,
    old_content: str,
    new_content: str,
) -> str:
    """
    Crée un aperçu unifié des différences.
    """

    old_lines = old_content.splitlines(
        keepends=True
    )

    new_lines = new_content.splitlines(
        keepends=True
    )

    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{relative_path} avant",
        tofile=f"{relative_path} après",
        lineterm="",
    )

    diff_text = "\n".join(
        diff_lines
    )

    if not diff_text:
        return "Aucune différence détectée."

    if len(diff_text) > MAX_DIFF_CHARACTERS:
        return (
            diff_text[:MAX_DIFF_CHARACTERS]
            + "\n\n[APERÇU DU DIFF TRONQUÉ]"
        )

    return diff_text


def _validate_content_syntax(
    path: Path,
    content: str,
) -> dict[str, Any]:
    """
    Valide la syntaxe de certains formats.

    Formats validés :
    - Python ;
    - JSON et JSONL ;
    - TOML.

    Les autres formats texte restent autorisés sans validation
    syntaxique particulière.
    """

    extension = path.suffix.lower()

    try:
        if extension == ".py":
            compile(
                content,
                str(path),
                "exec",
            )

            return {
                "checked": True,
                "valid": True,
                "message": "Syntaxe Python valide.",
            }

        if extension == ".json":
            json.loads(content)

            return {
                "checked": True,
                "valid": True,
                "message": "Syntaxe JSON valide.",
            }

        if extension == ".jsonl":
            for line_number, line in enumerate(
                content.splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    json.loads(line)

                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Ligne JSONL {line_number} invalide : "
                        f"{error}"
                    ) from error

            return {
                "checked": True,
                "valid": True,
                "message": "Syntaxe JSONL valide.",
            }

        if extension == ".toml":
            tomllib.loads(content)

            return {
                "checked": True,
                "valid": True,
                "message": "Syntaxe TOML valide.",
            }

    except (
        SyntaxError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        ValueError,
    ) as error:
        return {
            "checked": True,
            "valid": False,
            "message": str(error),
        }

    return {
        "checked": False,
        "valid": True,
        "message": (
            "Aucune validation syntaxique automatique "
            "n'est définie pour ce format."
        ),
    }


def _create_backup(
    path: Path,
) -> Path | None:
    """
    Crée une sauvegarde horodatée du fichier existant.
    """

    if not path.exists():
        return None

    relative_path = path.relative_to(
        WORKSPACE_ROOT
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S.%fZ")

    backup_path = (
        BACKUP_ROOT
        / relative_path.parent
        / f"{relative_path.name}.{timestamp}.bak"
    )

    backup_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        shutil.copy2(
            path,
            backup_path,
        )

    except OSError as error:
        raise FileModificationError(
            f"Impossible de créer la sauvegarde : {error}"
        ) from error

    return backup_path


def _tool_result(
    *,
    ok: bool,
    created: bool,
    message: str,
    error_code: str | None = None,
    proposal_id: str | None = None,
    path: str | None = None,
    operation: str | None = None,
) -> str:
    """Retourne un resultat structure et une action suivante explicite."""
    return json.dumps(
        {
            "ok": ok,
            "created": created,
            "proposal_id": proposal_id,
            "path": path,
            "operation": operation,
            "error_code": error_code,
            "message": message,
            "next_action": (
                "Informer l'utilisateur que la proposition attend son approbation, puis terminer."
                if created
                else "Corriger au plus une fois la cause ; sinon expliquer le blocage et terminer."
            ),
        },
        ensure_ascii=False,
    )


# -------------------------------------------------------------------
# Gestion générique des propositions
# -------------------------------------------------------------------

def _register_proposal(
    *,
    path: Path,
    operation: str,
    justification: str,
    old_content: str,
    new_content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Enregistre une proposition et retourne un message pour le modèle.
    """

    if not justification or not justification.strip():
        return (
            "Proposition refusée : une justification "
            "est obligatoire."
        )

    if old_content == new_content:
        return (
            "Aucune proposition créée : le contenu obtenu "
            "est identique au contenu actuel."
        )

    if len(new_content) > MAX_WRITE_CHARACTERS:
        return (
            "Proposition refusée : le contenu final dépasse "
            f"{MAX_WRITE_CHARACTERS} caractères."
        )

    relative_path = str(
        path.relative_to(WORKSPACE_ROOT)
    )

    syntax_validation = _validate_content_syntax(
        path=path,
        content=new_content,
    )

    # Une proposition syntaxiquement invalide ne doit JAMAIS atteindre
    # l'interface : elle y restait inapprouvable et donnait l'impression d'un
    # bug de l'application. On la refuse à la source, avec la cause la plus
    # probable (contenu coupé par la limite de tokens) et la marche à suivre.
    if syntax_validation.get("checked") and not syntax_validation.get("valid", True):
        return _tool_result(
            ok=False,
            created=False,
            path=relative_path,
            operation=operation,
            error_code="syntaxe_invalide",
            message=(
                "Aucune proposition créée : le contenu final est syntaxiquement "
                f"invalide. Détail : {syntax_validation['message']}. "
                "Cause la plus fréquente : contenu tronqué par la limite de "
                "tokens lors d'une réécriture complète. Utilise un patch ciblé "
                "(proposer_remplacement_cible) sur la seule zone à changer."
            ),
        )

    signature = _content_hash(
        f"{relative_path}\0{operation}\0{old_content}\0{new_content}"
    )
    with _PROPOSAL_LOCK:
        for existing in _PENDING_PROPOSALS.values():
            if (
                existing.get("status") == "pending"
                and existing.get("signature") == signature
            ):
                return _tool_result(
                    ok=True,
                    created=True,
                    proposal_id=existing.get("proposal_id"),
                    path=relative_path,
                    operation=operation,
                    message="Une proposition identique est deja en attente.",
                )
    proposal_id = uuid.uuid4().hex

    proposal = {
        "signature": signature,
        "proposal_id": proposal_id,
        "path": relative_path,
        "absolute_path": str(path),
        "operation": operation,
        "justification": justification.strip(),
        "old_content": old_content,
        "new_content": new_content,
        "old_content_hash": _content_hash(
            old_content
        ),
        "diff": _create_diff(
            relative_path=relative_path,
            old_content=old_content,
            new_content=new_content,
        ),
        "syntax_validation": syntax_validation,
        "metadata": metadata or {},
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "pending",
    }

    with _PROPOSAL_LOCK:
        _PENDING_PROPOSALS[
            proposal_id
        ] = proposal
        _save_proposals_unlocked()

    validation_message = syntax_validation[
        "message"
    ]

    validation_status = (
        "valide"
        if syntax_validation["valid"]
        else "invalide"
    )

    return _tool_result(
        ok=True,
        created=True,
        proposal_id=proposal_id,
        path=relative_path,
        operation=operation,
        message=(
            "Proposition créée, mais non appliquée. "
            f"Validation syntaxique {validation_status} : {validation_message}"
        ),
    )

def _get_proposal(
    proposal_id: str,
) -> dict[str, Any]:
    """
    Récupère une proposition à partir de son identifiant.
    """

    with _PROPOSAL_LOCK:
        proposal = _PENDING_PROPOSALS.get(
            proposal_id
        )

        if proposal is None:
            raise ProposalNotFoundError(
                "La proposition n'existe pas ou "
                "n'est plus disponible."
            )

        return deepcopy(
            proposal
        )


def _resolve_target_file(
    chemin: str,
) -> Path:
    """
    Résout et valide un chemin de fichier du workspace.
    """

    try:
        path = resolve_workspace_path(
            chemin
        )

    except ValueError as error:
        raise FileModificationError(
            str(error)
        ) from error

    _validate_writable_file(
        path
    )

    return path


# -------------------------------------------------------------------
# Outil 1 : création ou remplacement complet
# -------------------------------------------------------------------

@tool
def proposer_modification_fichier(
    chemin: str,
    nouveau_contenu: str,
    justification: str,
    autoriser_reduction_massive: bool = False,
) -> str:
    """
    Propose la création ou le remplacement complet d'un fichier texte.

    Utilise cet outil pour :
    - créer un fichier ;
    - remplacer un petit fichier ;
    - réécrire volontairement tout un fichier.

    Pour modifier seulement quelques lignes d'un fichier existant,
    préfère proposer_remplacement_cible ou proposer_insertion_ciblee.

    Le fichier n'est jamais écrit sans approbation dans Streamlit.
    """

    try:
        path = _resolve_target_file(
            chemin
        )

        old_content = _read_existing_content(
            path
        )

    except FileModificationError as error:
        return f"Proposition refusée : {error}"

    operation = (
        "creation_complete"
        if not path.exists()
        else "remplacement_complet"
    )

    if old_content and not autoriser_reduction_massive:
        removed_ratio = (
            1.0
            - (
                len(nouveau_contenu)
                / max(len(old_content), 1)
            )
        )

        if (
            removed_ratio
            > MAX_AUTOMATIC_REDUCTION_RATIO
        ):
            return (
                "Proposition refusée : le nouveau contenu "
                f"réduit le fichier de {removed_ratio:.1%}. "
                "Utilise un remplacement ciblé, ou rappelle "
                "cet outil avec autoriser_reduction_massive=True "
                "si cette réécriture complète est intentionnelle."
            )

    return _register_proposal(
        path=path,
        operation=operation,
        justification=justification,
        old_content=old_content,
        new_content=nouveau_contenu,
        metadata={
            "patch_type": "full_content",
            "allow_massive_reduction": (
                autoriser_reduction_massive
            ),
        },
    )


# -------------------------------------------------------------------
# Outil 2 : remplacement ciblé
# -------------------------------------------------------------------

@tool
def proposer_remplacement_cible(
    chemin: str,
    ancien_texte: str,
    nouveau_texte: str,
    justification: str,
) -> str:
    """
    Propose le remplacement exact d'un bloc dans un fichier existant.

    L'ancien texte doit apparaître exactement une fois dans le fichier.
    Cette méthode est recommandée pour corriger une fonction, un import,
    une configuration ou quelques lignes sans réécrire tout le fichier.

    Le fichier n'est jamais modifié sans approbation dans Streamlit.
    """

    if not ancien_texte:
        return (
            "Proposition refusée : ancien_texte "
            "ne peut pas être vide."
        )

    if len(ancien_texte) > MAX_PATCH_ARGUMENT_CHARACTERS:
        return (
            "Proposition refusée : ancien_texte dépasse "
            f"{MAX_PATCH_ARGUMENT_CHARACTERS} caractères."
        )

    if len(nouveau_texte) > MAX_PATCH_ARGUMENT_CHARACTERS:
        return (
            "Proposition refusée : nouveau_texte dépasse "
            f"{MAX_PATCH_ARGUMENT_CHARACTERS} caractères."
        )

    try:
        path = _resolve_target_file(
            chemin
        )

        if not path.exists():
            return (
                "Proposition refusée : le fichier n'existe pas. "
                "Utilise proposer_modification_fichier pour le créer."
            )

        old_content = _read_existing_content(
            path
        )

    except FileModificationError as error:
        return f"Proposition refusée : {error}"

    occurrence_count = old_content.count(
        ancien_texte
    )

    if occurrence_count == 0:
        return (
            "Proposition refusée : le texte original "
            "n'a pas été trouvé dans le fichier. "
            "Relis la zone concernée avant de proposer le patch."
        )

    if occurrence_count > 1:
        return (
            "Proposition refusée : le texte original apparaît "
            f"{occurrence_count} fois. Utilise une ancre plus précise "
            "afin de cibler une seule occurrence."
        )

    new_content = old_content.replace(
        ancien_texte,
        nouveau_texte,
        1,
    )

    return _register_proposal(
        path=path,
        operation="remplacement_cible",
        justification=justification,
        old_content=old_content,
        new_content=new_content,
        metadata={
            "patch_type": "replace",
            "old_text": ancien_texte,
            "new_text": nouveau_texte,
        },
    )


# -------------------------------------------------------------------
# Outil 3 : insertion ciblée
# -------------------------------------------------------------------

@tool
def proposer_insertion_ciblee(
    chemin: str,
    ancre: str,
    contenu_a_inserer: str,
    justification: str,
    position: str = "apres",
) -> str:
    """
    Propose une insertion avant ou après une ancre exacte.

    L'ancre doit apparaître exactement une fois dans le fichier.
    Le paramètre position accepte "avant" ou "apres".

    Cet outil est adapté pour ajouter :
    - un import ;
    - une fonction ;
    - un bloc de configuration ;
    - une entrée dans un fichier texte.

    Le fichier n'est jamais modifié sans approbation dans Streamlit.
    """

    if not ancre:
        return (
            "Proposition refusée : l'ancre ne peut pas être vide."
        )

    if not contenu_a_inserer:
        return (
            "Proposition refusée : le contenu à insérer "
            "ne peut pas être vide."
        )

    normalized_position = (
        position.strip().casefold()
    )

    if normalized_position not in {
        "avant",
        "apres",
    }:
        return (
            "Proposition refusée : position doit valoir "
            "'avant' ou 'apres'."
        )

    try:
        path = _resolve_target_file(
            chemin
        )

        if not path.exists():
            return (
                "Proposition refusée : le fichier n'existe pas."
            )

        old_content = _read_existing_content(
            path
        )

    except FileModificationError as error:
        return f"Proposition refusée : {error}"

    occurrence_count = old_content.count(
        ancre
    )

    if occurrence_count == 0:
        return (
            "Proposition refusée : l'ancre n'a pas été trouvée."
        )

    if occurrence_count > 1:
        return (
            "Proposition refusée : l'ancre apparaît "
            f"{occurrence_count} fois. Utilise une ancre plus précise."
        )

    if normalized_position == "avant":
        replacement = (
            contenu_a_inserer
            + ancre
        )
    else:
        replacement = (
            ancre
            + contenu_a_inserer
        )

    new_content = old_content.replace(
        ancre,
        replacement,
        1,
    )

    return _register_proposal(
        path=path,
        operation=f"insertion_{normalized_position}",
        justification=justification,
        old_content=old_content,
        new_content=new_content,
        metadata={
            "patch_type": "insert",
            "anchor": ancre,
            "inserted_content": contenu_a_inserer,
            "position": normalized_position,
        },
    )


# -------------------------------------------------------------------
# Outil 4 : suppression ciblée de contenu
# -------------------------------------------------------------------

@tool
def proposer_suppression_ciblee(
    chemin: str,
    texte_a_supprimer: str,
    justification: str,
) -> str:
    """
    Propose la suppression exacte d'un bloc de texte.

    Le bloc doit apparaître exactement une fois dans le fichier.
    Cet outil ne supprime jamais le fichier lui-même.

    Le changement n'est jamais appliqué sans approbation Streamlit.
    """

    if not texte_a_supprimer:
        return (
            "Proposition refusée : le texte à supprimer "
            "ne peut pas être vide."
        )

    try:
        path = _resolve_target_file(
            chemin
        )

        if not path.exists():
            return (
                "Proposition refusée : le fichier n'existe pas."
            )

        old_content = _read_existing_content(
            path
        )

    except FileModificationError as error:
        return f"Proposition refusée : {error}"

    occurrence_count = old_content.count(
        texte_a_supprimer
    )

    if occurrence_count == 0:
        return (
            "Proposition refusée : le texte à supprimer "
            "n'a pas été trouvé."
        )

    if occurrence_count > 1:
        return (
            "Proposition refusée : le texte à supprimer apparaît "
            f"{occurrence_count} fois. Utilise un bloc plus précis."
        )

    new_content = old_content.replace(
        texte_a_supprimer,
        "",
        1,
    )

    return _register_proposal(
        path=path,
        operation="suppression_ciblee",
        justification=justification,
        old_content=old_content,
        new_content=new_content,
        metadata={
            "patch_type": "delete",
            "deleted_text": texte_a_supprimer,
        },
    )


# -------------------------------------------------------------------
# Fonctions réservées à Streamlit
# -------------------------------------------------------------------

def get_pending_proposals() -> list[dict[str, Any]]:
    """
    Retourne toutes les propositions en attente.
    """

    with _PROPOSAL_LOCK:
        proposals = [
            deepcopy(proposal)
            for proposal in _PENDING_PROPOSALS.values()
            if proposal.get("status") == "pending"
        ]

    return sorted(
        proposals,
        key=lambda proposal: proposal[
            "created_at"
        ],
    )


def get_proposal_statuses(proposal_ids: list[str]) -> dict[str, str]:
    """Retourne les états courants sans exposer le contenu des propositions."""
    with _PROPOSAL_LOCK:
        statuses: dict[str, str] = {}
        for proposal_id in proposal_ids:
            if proposal_id in _PENDING_PROPOSALS:
                statuses[proposal_id] = str(
                    _PENDING_PROPOSALS[proposal_id].get("status", "unknown")
                )
            elif proposal_id in _RESOLVED_PROPOSAL_STATUSES:
                statuses[proposal_id] = _RESOLVED_PROPOSAL_STATUSES[proposal_id]
        return statuses


def _reapply_proposal_on_current_content(
    *,
    proposal: dict[str, Any],
    current_content: str,
) -> str:
    """
    Réapplique un patch ciblé sur le contenu actuel d'un fichier.

    Cette fonction est utilisée lorsque le fichier a changé depuis la
    création de la proposition (par exemple parce qu'une autre
    proposition ciblée sur le même fichier vient d'être approuvée).

    Elle recalcule le contenu final à partir du contenu ACTUEL, sans
    faire confiance au snapshot figé au moment de la proposition.

    Les patchs fusionnables sont :
    - remplacement ciblé ;
    - insertion avant / après une ancre ;
    - suppression ciblée.

    Un remplacement ou une création complète n'est pas fusionnable en
    toute sécurité : dans ce cas, on refuse l'application afin d'éviter
    d'écraser des modifications concurrentes.
    """

    operation = proposal.get(
        "operation"
    )

    metadata = proposal.get(
        "metadata",
        {},
    )

    patch_type = metadata.get(
        "patch_type"
    )

    if operation == "remplacement_cible":
        ancien_texte = metadata.get(
            "old_text"
        )

        nouveau_texte = metadata.get(
            "new_text"
        )

        if not ancien_texte:
            raise ConcurrentModificationError(
                "Impossible de réappliquer le remplacement ciblé : "
                "le texte d'origine n'est pas disponible dans la "
                "proposition."
            )

        occurrence_count = current_content.count(
            ancien_texte
        )

        if occurrence_count == 0:
            raise ConcurrentModificationError(
                "Le fichier a changé et le texte à remplacer n'y "
                "figure plus. Régénère la proposition sur la base "
                "du fichier à jour."
            )

        if occurrence_count > 1:
            raise ConcurrentModificationError(
                "Le fichier a changé et le texte à remplacer y "
                "apparaît désormais plusieurs fois. Régénère la "
                "proposition avec un bloc plus précis."
            )

        return current_content.replace(
            ancien_texte,
            nouveau_texte or "",
            1,
        )

    if operation in (
        "insertion_avant",
        "insertion_apres",
    ):
        ancre = metadata.get(
            "anchor"
        )

        contenu_a_inserer = metadata.get(
            "inserted_content"
        )

        if not ancre:
            raise ConcurrentModificationError(
                "Impossible de réappliquer l'insertion : l'ancre "
                "n'est pas disponible dans la proposition."
            )
        if contenu_a_inserer is None:
            raise ConcurrentModificationError(
                "Impossible de réappliquer l'insertion : le contenu "
                "à insérer n'est pas disponible dans la proposition."
            )

        occurrence_count = current_content.count(
            ancre
        )

        if occurrence_count == 0:
            raise ConcurrentModificationError(
                "Le fichier a changé et l'ancre d'insertion n'y "
                "figure plus. Régénère la proposition sur la base "
                "du fichier à jour."
            )

        if occurrence_count > 1:
            raise ConcurrentModificationError(
                "Le fichier a changé et l'ancre d'insertion y "
                "apparaît désormais plusieurs fois. Régénère la "
                "proposition avec une ancre plus précise."
            )

        if operation == "insertion_avant":
            replacement = (
                f"{contenu_a_inserer or ''}{ancre}"
            )

        else:
            replacement = (
                f"{ancre}{contenu_a_inserer or ''}"
            )

        return current_content.replace(
            ancre,
            replacement,
            1,
        )

    if operation == "suppression_ciblee":
        texte_a_supprimer = metadata.get(
            "deleted_text"
        )

        if not texte_a_supprimer:
            raise ConcurrentModificationError(
                "Impossible de réappliquer la suppression : le "
                "texte à supprimer n'est pas disponible dans la "
                "proposition."
            )

        occurrence_count = current_content.count(
            texte_a_supprimer
        )

        if occurrence_count == 0:
            raise ConcurrentModificationError(
                "Le fichier a changé et le texte à supprimer n'y "
                "figure plus. Régénère la proposition sur la base "
                "du fichier à jour."
            )

        if occurrence_count > 1:
            raise ConcurrentModificationError(
                "Le fichier a changé et le texte à supprimer y "
                "apparaît désormais plusieurs fois. Régénère la "
                "proposition avec un bloc plus précis."
            )

        return current_content.replace(
            texte_a_supprimer,
            "",
            1,
        )

    # creation_complete / remplacement_complet / patch_type full_content
    raise ConcurrentModificationError(
        "Le fichier a été modifié depuis la création de cette "
        "proposition et il s'agit d'une réécriture complète, qui ne "
        "peut pas être fusionnée automatiquement en toute sécurité. "
        "Régénère la proposition sur la base du fichier à jour."
    )


def approve_proposal(
    proposal_id: str,
) -> dict[str, Any]:
    """
    Applique une proposition après approbation utilisateur.

    Vérifications effectuées :
    - statut de la proposition ;
    - fichier toujours dans le workspace ;
    - contenu actuel identique au snapshot ;
    - validation syntaxique réussie ;
    - sauvegarde de l'ancienne version ;
    - écriture atomique.
    """

    proposal = _get_proposal(
        proposal_id
    )

    if proposal.get("status") != "pending":
        raise FileModificationError(
            "La proposition n'est plus en attente."
        )

    syntax_validation = proposal.get(
        "syntax_validation",
        {},
    )

    if not syntax_validation.get(
        "valid",
        True,
    ):
        raise FileModificationError(
            "La proposition contient une erreur de syntaxe : "
            f"{syntax_validation.get('message', 'erreur inconnue')}"
        )

    path = _resolve_target_file(
        proposal["path"]
    )

    current_content = _read_existing_content(
        path
    )

    current_hash = _content_hash(
        current_content
    )

    expected_hash = proposal[
        "old_content_hash"
    ]

    # Si le fichier n'a pas changé depuis la création de la
    # proposition, on applique directement le nouveau contenu.
    # Sinon, on tente de réappliquer le patch ciblé sur le contenu
    # actuel. Cela permet d'approuver successivement plusieurs
    # propositions indépendantes sur un même fichier sans conflit.
    if current_hash == expected_hash:
        content_to_write = proposal["new_content"]

    else:
        content_to_write = _reapply_proposal_on_current_content(
            proposal=proposal,
            current_content=current_content,
        )

    final_validation = _validate_content_syntax(
        path=path,
        content=content_to_write,
    )
    if not final_validation.get("valid", True):
        raise FileModificationError(
            "Le contenu final contient une erreur de syntaxe : "
            f"{final_validation.get('message', 'erreur inconnue')}"
        )

    backup_path = _create_backup(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        temporary_path.write_text(
            content_to_write,
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    except OSError as error:
        try:
            temporary_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise FileModificationError(
            f"Impossible d'appliquer la modification : {error}"
        ) from error

    with _PROPOSAL_LOCK:
        stored_proposal = _PENDING_PROPOSALS[
            proposal_id
        ]

        stored_proposal["status"] = "approved"

        stored_proposal["applied_at"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        stored_proposal["backup_path"] = (
            str(
                backup_path.relative_to(
                    PROJECT_ROOT
                )
            )
            if backup_path is not None
            else None
        )
        stored_proposal["applied_content_hash"] = _content_hash(
            content_to_write
        )
        _save_proposals_unlocked()

        return deepcopy(
            stored_proposal
        )


def reject_proposal(
    proposal_id: str,
) -> dict[str, Any]:
    """
    Refuse une proposition sans modifier le fichier.
    """

    with _PROPOSAL_LOCK:
        proposal = _PENDING_PROPOSALS.get(
            proposal_id
        )

        if proposal is None:
            raise ProposalNotFoundError(
                "La proposition n'existe pas ou "
                "n'est plus disponible."
            )

        if proposal.get("status") != "pending":
            raise FileModificationError(
                "La proposition n'est plus en attente."
            )

        proposal["status"] = "rejected"

        proposal["rejected_at"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
        _save_proposals_unlocked()

        return deepcopy(
            proposal
        )


def approve_pending_proposals(
    proposal_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Applique atomiquement plusieurs propositions après prévalidation."""
    with _PROPOSAL_LOCK:
        return _approve_pending_proposals_unlocked(proposal_ids)


def _approve_pending_proposals_unlocked(
    proposal_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Implémentation transactionnelle ; verrou de propositions déjà acquis."""
    pending = get_pending_proposals()
    requested = set(proposal_ids) if proposal_ids is not None else None
    selected = [
        proposal for proposal in pending
        if requested is None or str(proposal["proposal_id"]) in requested
    ]
    if not selected:
        return []

    transaction_id = uuid.uuid4().hex
    virtual_contents: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    originally_existed: dict[Path, bool] = {}
    final_by_proposal: dict[str, str] = {}
    try:
        for proposal in selected:
            path = _resolve_target_file(proposal["path"])
            if path not in originals:
                originally_existed[path] = path.exists()
                originals[path] = _read_existing_content(path)
                virtual_contents[path] = originals[path]
            current_content = virtual_contents[path]
            if _content_hash(current_content) == proposal["old_content_hash"]:
                final_content = proposal["new_content"]
            else:
                final_content = _reapply_proposal_on_current_content(
                    proposal=proposal,
                    current_content=current_content,
                )
            validation = _validate_content_syntax(path=path, content=final_content)
            if not validation.get("valid", True):
                raise FileModificationError(
                    f"{proposal['path']} : {validation.get('message', 'syntaxe invalide')}"
                )
            virtual_contents[path] = final_content
            final_by_proposal[str(proposal["proposal_id"])] = final_content
    except FileModificationError as error:
        return [{
            "proposal_id": str(proposal["proposal_id"]),
            "path": proposal.get("path"),
            "ok": False,
            "error": f"Transaction annulée avant écriture : {error}",
            "change_set_id": transaction_id,
        } for proposal in selected]

    backups: dict[Path, Path | None] = {}
    temporary_paths: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for path, content in virtual_contents.items():
            backups[path] = _create_backup(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{transaction_id}.tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary_paths[path] = temporary
        for path, temporary in temporary_paths.items():
            temporary.replace(path)
            committed.append(path)
    except OSError as error:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        rollback_errors: list[str] = []
        for path in committed:
            try:
                original = originals[path]
                if originally_existed[path]:
                    path.write_text(original, encoding="utf-8")
                else:
                    path.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(f"{path}: {rollback_error}")
        detail = f"Transaction annulée pendant l'écriture : {error}"
        if rollback_errors:
            detail += " ; rollback incomplet : " + " | ".join(rollback_errors)
        return [{
            "proposal_id": str(proposal["proposal_id"]),
            "path": proposal.get("path"),
            "ok": False,
            "error": detail,
            "change_set_id": transaction_id,
        } for proposal in selected]

    applied_at = datetime.now(timezone.utc).isoformat()
    with _PROPOSAL_LOCK:
        for proposal in selected:
            proposal_id = str(proposal["proposal_id"])
            stored = _PENDING_PROPOSALS[proposal_id]
            stored["status"] = "approved"
            stored["applied_at"] = applied_at
            stored["change_set_id"] = transaction_id
            stored["applied_content_hash"] = _content_hash(
                final_by_proposal[proposal_id]
            )
        _save_proposals_unlocked()
    return [{
        "proposal_id": str(proposal["proposal_id"]),
        "path": proposal.get("path"),
        "ok": True,
        "result": deepcopy(_PENDING_PROPOSALS[str(proposal["proposal_id"])]),
        "change_set_id": transaction_id,
    } for proposal in selected]


def clear_resolved_proposals() -> None:
    """
    Retire de la mémoire les propositions traitées.

    Les fichiers et sauvegardes ne sont jamais supprimés.
    """

    with _PROPOSAL_LOCK:
        resolved_ids = [
            proposal_id
            for proposal_id, proposal
            in _PENDING_PROPOSALS.items()
            if proposal.get("status") != "pending"
        ]

        for proposal_id in resolved_ids:
            _RESOLVED_PROPOSAL_STATUSES[proposal_id] = str(
                _PENDING_PROPOSALS[proposal_id].get("status", "unknown")
            )
            _PENDING_PROPOSALS.pop(
                proposal_id,
                None,
            )
        _save_proposals_unlocked()


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Ecriture directe réservée aux workers du mode multi-agent
# -------------------------------------------------------------------

def _apply_direct_change(tool_object: Any, arguments: dict[str, Any]) -> str:
    """Valide, sauvegarde et applique une écriture sans étape d'approbation UI."""
    path = resolve_workspace_path(str(arguments.get("chemin", "")))
    with path_lock(path):
        session = CURRENT_EXECUTION.get()
        operation = session.operation_key(tool_object.name, arguments) if session else ""
        if session:
            previous = session.replay(operation, path)
            if previous is not None:
                return json.dumps(previous, ensure_ascii=False)
            expected = session.read_versions.get(str(path))
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if expected is not None and actual != expected:
                return json.dumps({"ok": False, "applied": False, "path": arguments["chemin"],
                                   "error": "Fichier modifié depuis la lecture ; relire la zone avant de réessayer."}, ensure_ascii=False)
        result = json.loads(_apply_direct_change_unlocked(tool_object, arguments))
        if result.get("applied") is True:
            result["content_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
            result["completed_at_ns"] = time.time_ns()
            if session:
                session.observe(path)
                session.record_write(operation, result)
        return json.dumps(result, ensure_ascii=False)


def _apply_direct_change_unlocked(tool_object: Any, arguments: dict[str, Any]) -> str:
    raw_result = tool_object.invoke(arguments)
    try:
        prepared = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        return str(raw_result)

    proposal_id = prepared.get("proposal_id")
    if not prepared.get("ok") or not prepared.get("created") or not proposal_id:
        return str(raw_result)

    try:
        applied = approve_proposal(str(proposal_id))
    except Exception as error:
        # Une écriture refusée ne doit pas rester comme proposition en attente :
        # le mode multi-agent n'utilise plus ce circuit d'approbation.
        try:
            reject_proposal(str(proposal_id))
        except Exception:
            pass
        return json.dumps({
            "ok": False,
            "applied": False,
            "path": prepared.get("path"),
            "error": f"{type(error).__name__}: {error}",
        }, ensure_ascii=False)

    return json.dumps({
        "ok": True,
        "created": True,
        "applied": True,
        "proposal_id": str(proposal_id),
        "path": applied.get("path"),
        "operation": applied.get("operation"),
        "backup_path": applied.get("backup_path"),
        "changed": True,
        "message": "Modification écrite et sauvegardée.",
    }, ensure_ascii=False)


@tool
def ecrire_fichier_directement(
    chemin: str,
    nouveau_contenu: str,
    justification: str = "Modification demandée",
    autoriser_reduction_massive: bool = False,
) -> str:
    """Crée ou remplace directement un fichier texte du workspace.

    La modification est appliquée immédiatement après validation syntaxique
    et sauvegarde de l'ancienne version. Préfère les outils de patch ciblé
    pour limiter la sortie générée et éviter de réécrire de gros fichiers.
    """
    return _apply_direct_change(
        proposer_modification_fichier,
        {
            "chemin": chemin,
            "nouveau_contenu": nouveau_contenu,
            "justification": justification,
            "autoriser_reduction_massive": autoriser_reduction_massive,
        },
    )


@tool
def remplacer_bloc_directement(
    chemin: str,
    ancien_texte: str,
    nouveau_texte: str,
    justification: str = "Modification demandée",
) -> str:
    """Remplace une occurrence exacte dans un fichier du workspace et applique le changement."""
    return _apply_direct_change(
        proposer_remplacement_cible,
        {
            "chemin": chemin,
            "ancien_texte": ancien_texte,
            "nouveau_texte": nouveau_texte,
            "justification": justification,
        },
    )


@tool
def inserer_bloc_directement(
    chemin: str,
    ancre: str,
    contenu_a_inserer: str,
    justification: str = "Modification demandée",
    position: str = "apres",
) -> str:
    """Insère un bloc autour d'une ancre unique et applique le changement."""
    return _apply_direct_change(
        proposer_insertion_ciblee,
        {
            "chemin": chemin,
            "ancre": ancre,
            "contenu_a_inserer": contenu_a_inserer,
            "justification": justification,
            "position": position,
        },
    )


@tool
def supprimer_bloc_directement(
    chemin: str,
    texte_a_supprimer: str,
    justification: str = "Modification demandée",
) -> str:
    """Supprime une occurrence exacte d'un bloc et applique le changement."""
    return _apply_direct_change(
        proposer_suppression_ciblee,
        {
            "chemin": chemin,
            "texte_a_supprimer": texte_a_supprimer,
            "justification": justification,
        },
    )


@tool
def appliquer_modifications_directement(modifications: list[dict[str, str]]) -> str:
    """Applique ensemble des changements liés sur plusieurs fichiers.

    Chaque élément contient chemin et soit contenu, soit ancien_texte/nouveau_texte.
    Un seul changement par chemin. Tous sont prévalidés avant écriture ; sauvegardes
    et restauration en cas d'erreur d'application. Aucun rapport/justification requis.
    """
    if not modifications or len(modifications) > 20:
        return json.dumps({"ok": False, "error": "Fournir de 1 à 20 modifications."})
    try:
        paths = [resolve_workspace_path(item["chemin"]) for item in modifications]
        if len(set(paths)) != len(paths):
            raise ValueError("Un seul changement par fichier dans un lot.")
    except (KeyError, TypeError, ValueError) as error:
        return json.dumps({"ok": False, "error": str(error)})
    session = CURRENT_EXECUTION.get()
    operation = session.operation_key("batch", {"modifications": modifications}) if session else ""
    prepared: list[str] = []
    with ExitStack() as stack:
        for path in sorted(paths):
            stack.enter_context(path_lock(path))
        if session:
            replayed = [session.replay(operation + str(path), path) for path in paths]
            if all(replayed):
                return json.dumps({"ok": True, "files": replayed}, ensure_ascii=False)
        try:
            for path, item in zip(paths, modifications):
                if session and str(path) in session.read_versions:
                    current = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                    if session.read_versions[str(path)] != current:
                        raise ConcurrentModificationError(f"Relire {item['chemin']} : fichier modifié depuis la lecture.")
                if "ancien_texte" in item:
                    raw = proposer_remplacement_cible.invoke({"chemin": item["chemin"],
                          "ancien_texte": item["ancien_texte"], "nouveau_texte": item["nouveau_texte"],
                          "justification": "Modification groupée demandée"})
                else:
                    raw = proposer_modification_fichier.invoke({"chemin": item["chemin"],
                          "nouveau_contenu": item["contenu"], "justification": "Modification groupée demandée"})
                result = json.loads(raw)
                if not result.get("ok") or not result.get("created"):
                    raise FileModificationError(result.get("message") or result.get("error") or "Prévalidation refusée.")
                prepared.append(result["proposal_id"])
            results = approve_pending_proposals(prepared)
            if not all(result.get("ok") for result in results):
                raise FileModificationError(str(results[0].get("error", "Lot non appliqué.")))
        except (FileModificationError, KeyError, ValueError, TypeError, OSError) as error:
            for proposal_id in prepared:
                try:
                    reject_proposal(proposal_id)
                except FileModificationError:
                    pass
            return json.dumps({"ok": False, "files": [], "error": str(error)}, ensure_ascii=False)
        files = []
        for path, item, result in zip(paths, modifications, results):
            record = {"ok": True, "applied": True, "path": item["chemin"],
                      "proposal_id": result["proposal_id"],
                      "backup_path": result.get("result", {}).get("backup_path"),
                      "content_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "completed_at_ns": time.time_ns()}
            if session:
                session.observe(path)
                session.record_write(operation + str(path), record)
            files.append(record)
        return json.dumps({"ok": True, "files": files}, ensure_ascii=False)


DIRECT_WRITE_TOOLS = [
    ecrire_fichier_directement,
    remplacer_bloc_directement,
    inserer_bloc_directement,
    supprimer_bloc_directement,
    appliquer_modifications_directement,
]


# Outils exportés vers l'agent
# -------------------------------------------------------------------

WRITE_PROPOSAL_TOOLS = [
    proposer_modification_fichier,
    proposer_remplacement_cible,
    proposer_insertion_ciblee,
    proposer_suppression_ciblee,
]
