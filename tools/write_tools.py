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
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

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
    payload = {"schema_version": 1, "proposals": list(_PENDING_PROPOSALS.values())}
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
            _PENDING_PROPOSALS.pop(
                proposal_id,
                None,
            )
        _save_proposals_unlocked()


# -------------------------------------------------------------------
# Outils exportés vers l'agent
# -------------------------------------------------------------------

WRITE_PROPOSAL_TOOLS = [
    proposer_modification_fichier,
    proposer_remplacement_cible,
    proposer_insertion_ciblee,
    proposer_suppression_ciblee,
]