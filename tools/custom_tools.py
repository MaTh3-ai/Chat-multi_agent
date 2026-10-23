"""Outils sécurisés de consultation du workspace.

Tous les chemins sont limités au dossier workspace. Ce module fournit des
outils non destructifs de consultation. Les propositions de modification
sont fournies par tools.write_tools et nécessitent une approbation humaine.
Le mode d'accès réel du workspace est déterminé dynamiquement.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from core.command_journal import record_command
from core.agent_runtime import CURRENT_EXECUTION, command_category, path_lock


# -------------------------------------------------------------------
# Détection des chemins du projet
# -------------------------------------------------------------------

def detect_project_root() -> Path:
    """
    Détermine automatiquement la racine du projet.

    La variable d'environnement APP_PROJECT_ROOT est prioritaire.

    Sinon, la fonction cherche un dossier contenant workspace
    parmi les parents de custom_tools.py.
    """

    configured_root = os.getenv(
        "APP_PROJECT_ROOT"
    )

    if configured_root:
        return Path(configured_root).expanduser().resolve()

    current_file = Path(__file__).resolve()
    repository_root = current_file.parent.parent
    sagemaker_root = Path("/home/sagemaker-user/app_chat")

    possible_roots = [
        current_file.parent,
        repository_root,
        sagemaker_root,
    ]

    for possible_root in possible_roots:
        if (possible_root / "workspace").is_dir():
            return possible_root.resolve()

    # Aucun dossier `workspace` existant : on retient le chemin SageMaker
    # seulement s'il est réellement utilisable. Sinon on reste dans le dépôt
    # courant, sans quoi l'import du module échoue sur toute autre machine
    # (tentative de création de /home/... hors de SageMaker).
    if sagemaker_root.parent.is_dir():
        return sagemaker_root.resolve()
    return repository_root.resolve()


PROJECT_ROOT = detect_project_root()

WORKSPACE_ROOT = Path(
    os.getenv(
        "APP_WORKSPACE_PATH",
        str(PROJECT_ROOT / "workspace"),
    )
).expanduser().resolve()


# Le dossier est créé s'il n'existe pas encore.
WORKSPACE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# -------------------------------------------------------------------
# Limites de sécurité et d'optimisation
# -------------------------------------------------------------------

# Lire par tranches trop petites multipliait les appels d'outils (et donc les
# tokens de contexte reinjectes a chaque tour) sans jamais donner assez de
# contexte pour terminer une tache. Les bornes ci-dessous restent protectrices.
MAX_READ_CHARACTERS = 40_000
DEFAULT_READ_CHARACTERS = 8_000

MAX_SEARCH_RESULTS = 30
MAX_LIST_RESULTS = 150

# Les très gros fichiers ne sont pas chargés entièrement en mémoire.
MAX_TEXT_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_COMMAND_TIMEOUT_SECONDS = 120
MAX_COMMAND_OUTPUT_CHARACTERS = 20_000
ALLOWED_COMMANDS = frozenset({"python", "python.exe", "python3", "pytest", "py", "ruff", "node", "npm", "pnpm", "yarn", "tsc", "git", "rg"})

ALLOWED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".log",
    ".sql",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".sh",
    ".ps1",
    ".env.example",
}

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}

IGNORED_FILE_NAMES = {
    ".DS_Store",
}


# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------

class WorkspaceSecurityError(ValueError):
    """Le chemin demandé sort du workspace autorisé."""


class UnsupportedFileTypeError(ValueError):
    """Le type du fichier demandé n'est pas pris en charge."""


# -------------------------------------------------------------------
# Fonctions internes
# -------------------------------------------------------------------

def resolve_workspace_path(
    relative_path: str,
) -> Path:
    """
    Transforme un chemin relatif en chemin sécurisé du workspace.

    Exemples autorisés :
    - .
    - data
    - data/fichier.csv

    Exemples refusés :
    - /etc/passwd
    - ../app.py
    - ../../autre_dossier
    """

    if relative_path is None:
        raise WorkspaceSecurityError(
            "Le chemin ne peut pas être nul."
        )

    cleaned_path = str(relative_path).strip()

    if not cleaned_path:
        cleaned_path = "."

    requested_path = Path(cleaned_path)

    if requested_path.is_absolute():
        raise WorkspaceSecurityError(
            "Les chemins absolus ne sont pas autorisés. "
            "Utilise un chemin relatif au workspace."
        )

    resolved_path = (
        WORKSPACE_ROOT / requested_path
    ).resolve()

    try:
        resolved_path.relative_to(
            WORKSPACE_ROOT
        )

    except ValueError as error:
        raise WorkspaceSecurityError(
            "Le chemin demandé sort du workspace autorisé."
        ) from error

    return resolved_path


def get_relative_path(path: Path) -> str:
    """
    Retourne le chemin d'un élément relativement au workspace.
    """

    relative_path = path.relative_to(
        WORKSPACE_ROOT
    )

    relative_text = str(relative_path)

    return (
        "."
        if relative_text == "."
        else relative_text
    )


def is_ignored_path(path: Path) -> bool:
    """
    Indique si un fichier ou dossier technique doit être ignoré.
    """

    if path.name in IGNORED_FILE_NAMES:
        return True

    relative_path = path.relative_to(
        WORKSPACE_ROOT
    )

    return any(
        part in IGNORED_DIRECTORY_NAMES
        for part in relative_path.parts
    )


def validate_text_file(path: Path) -> None:
    """
    Vérifie que le fichier peut être lu comme un fichier texte.
    """

    extension = path.suffix.lower()

    if extension not in ALLOWED_TEXT_EXTENSIONS:
        allowed_extensions = ", ".join(
            sorted(ALLOWED_TEXT_EXTENSIONS)
        )

        raise UnsupportedFileTypeError(
            f"Format non pris en charge : "
            f"{extension or 'sans extension'}. "
            f"Formats texte autorisés : "
            f"{allowed_extensions}"
        )


def format_file_size(size_bytes: int) -> str:
    """
    Formate une taille de fichier de manière lisible.
    """

    units = [
        "octets",
        "Ko",
        "Mo",
        "Go",
    ]

    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "octets":
                return f"{int(size)} {unit}"

            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size_bytes} octets"


def iterate_workspace_files(
    directory: Path,
):
    """
    Parcourt les fichiers du workspace en ignorant les dossiers
    techniques et les caches.
    """

    for path in sorted(
        directory.rglob("*"),
        key=lambda item: str(item).casefold(),
    ):
        if is_ignored_path(path):
            continue

        if path.is_file():
            yield path


def read_text_excerpt(
    path: Path,
    start: int,
    length: int,
) -> tuple[str, int, bool]:
    """
    Lit un extrait d'un fichier texte sans charger inutilement
    la totalité des fichiers très volumineux.

    Retourne :
    - le texte lu ;
    - la position de fin ;
    - un indicateur signalant s'il reste du texte.
    """

    safe_start = max(0, start)

    safe_length = max(
        1,
        min(length, MAX_READ_CHARACTERS),
    )

    session = CURRENT_EXECUTION.get()
    if session is not None:
        session.observe(path)
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        # On saute exactement safe_start caractères puis on lit
        # safe_length caractères. Contrairement à file.seek(), ce
        # découpage reste cohérent avec le comptage en caractères
        # utilisé pour la pagination, y compris avec des caractères
        # multi-octets (accents, emojis).
        file.read(safe_start)
        excerpt = file.read(safe_length)

        next_character = file.read(1)
        has_more_content = bool(next_character)

    end_position = safe_start + len(excerpt)

    return (
        excerpt,
        end_position,
        has_more_content,
    )


def check_workspace_writable() -> tuple[bool, str]:
    """Teste réellement création, lecture et suppression dans le workspace."""
    test_path = WORKSPACE_ROOT / f".workspace_write_test_{os.getpid()}"
    expected = "workspace write test\n"
    try:
        test_path.write_text(expected, encoding="utf-8")
        if test_path.read_text(encoding="utf-8") != expected:
            return False, "Le contenu du fichier temporaire n'a pas pu être vérifié."
        test_path.unlink()
        return True, "Création, lecture et suppression temporaires réussies."
    except OSError as error:
        try:
            test_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"{type(error).__name__}: {error}"



def workspace_snapshot() -> dict:
    """Retourne un diagnostic structure du workspace pour le code Python."""
    files = list(iterate_workspace_files(WORKSPACE_ROOT))
    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "exists": WORKSPACE_ROOT.exists(),
        "is_directory": WORKSPACE_ROOT.is_dir(),
        "file_count": len(files),
        "files": [get_relative_path(path) for path in files[:MAX_LIST_RESULTS]],
    }


def discover_file_candidates(
    terms: list[str],
    directory: str = ".",
    limit: int = 20,
) -> list[dict]:
    """Classe les fichiers candidats sans dépendre d'un format textuel."""
    needles = list(dict.fromkeys(
        str(term).strip().casefold() for term in terms if str(term).strip()
    ))
    if not needles:
        return []
    root = resolve_workspace_path(directory)
    if not root.is_dir():
        return []
    candidates = []
    for path in iterate_workspace_files(root):
        if path.is_symlink() or path.suffix.lower() not in ALLOWED_TEXT_EXTENSIONS:
            continue
        relative = get_relative_path(path)
        score = sum(8 for term in needles if term in relative.casefold())
        try:
            size = path.stat().st_size
            if size <= 500_000:
                sample = path.read_text(encoding="utf-8", errors="replace")[:12_000].casefold()
                score += sum(min(sample.count(term), 3) for term in needles)
        except OSError:
            continue
        if score:
            candidates.append({"path": relative, "score": score, "size": size})
    candidates.sort(key=lambda item: (-item["score"], item["path"].casefold()))
    return candidates[:max(1, min(int(limit), 40))]


def read_workspace_file(
    relative_path: str,
    start: int = 0,
    length: int = DEFAULT_READ_CHARACTERS,
) -> dict:
    """Lit un fichier et retourne un résultat structure, jamais un faux succès."""
    try:
        path = resolve_workspace_path(relative_path)
        if not path.exists():
            return {"ok": False, "path": relative_path, "error": "Fichier introuvable."}
        if not path.is_file():
            return {"ok": False, "path": relative_path, "error": "Le chemin n'est pas un fichier."}
        validate_text_file(path)
        size = path.stat().st_size
        if size > MAX_TEXT_FILE_SIZE_BYTES:
            return {"ok": False, "path": relative_path, "error": "Fichier trop volumineux."}
        excerpt, end, has_more = read_text_excerpt(path, start, length)
        if not excerpt:
            return {"ok": False, "path": relative_path, "error": "Aucun contenu lisible."}
        return {
            "ok": True,
            "path": get_relative_path(path),
            "content": excerpt,
            "start": max(0, start),
            "end": end,
            "has_more": has_more,
            "size": size,
        }
    except (WorkspaceSecurityError, UnsupportedFileTypeError, OSError) as error:
        return {"ok": False, "path": relative_path, "error": f"{type(error).__name__}: {error}"}

# -------------------------------------------------------------------
# Outil 1 : informations sur le workspace
# -------------------------------------------------------------------

@tool
def informations_workspace() -> str:
    """Retourne les informations et le mode d'accès réel du workspace.

    Un accès système en écriture ne donne pas au modèle un droit d'écriture
    directe. Toute modification passe par un outil proposer_* puis une
    approbation explicite dans Streamlit.
    """
    try:
        files = list(iterate_workspace_files(WORKSPACE_ROOT))
        directories = [
            path for path in WORKSPACE_ROOT.rglob("*")
            if path.is_dir() and not is_ignored_path(path)
        ]
        total_size = sum(path.stat().st_size for path in files)
    except OSError as error:
        return (
            f"Workspace : {WORKSPACE_ROOT}\n"
            f"Existe : {WORKSPACE_ROOT.exists()}\n"
            f"Erreur de lecture : {error}"
        )
    writable, details = check_workspace_writable()
    access_mode = "lecture et écriture" if writable else "lecture seule"
    proposal_status = (
        "disponibles avec approbation humaine obligatoire"
        if writable else "indisponibles tant que l'écriture système échoue"
    )
    return (
        f"Workspace : {WORKSPACE_ROOT}\n"
        f"Existe : {WORKSPACE_ROOT.exists()}\n"
        f"Est un dossier : {WORKSPACE_ROOT.is_dir()}\n"
        f"Nombre de dossiers : {len(directories)}\n"
        f"Nombre de fichiers : {len(files)}\n"
        f"Taille totale : {format_file_size(total_size)}\n"
        f"Mode d'accès réel : {access_mode}\n"
        f"Diagnostic d'écriture : {details}\n"
        f"Outils de proposition : {proposal_status}\n"
        "Écriture directe par le modèle : interdite\n"
        f"Écriture après approbation : {'possible' if writable else 'impossible'}"
    )


# -------------------------------------------------------------------
# Outil 2 : lister les fichiers
# -------------------------------------------------------------------

@tool
def lister_fichiers(
    dossier: str = ".",
    profondeur_maximale: int = 3,
) -> str:
    """
    Liste les dossiers et fichiers présents dans le workspace.

    Le paramètre dossier est relatif au workspace.

    Exemples :
    - "." pour la racine ;
    - "data" pour le dossier workspace/data.

    La profondeur maximale est limitée à 5.
    """

    try:
        directory = resolve_workspace_path(
            dossier
        )

    except WorkspaceSecurityError as error:
        return f"Accès refusé : {error}"

    if not directory.exists():
        return (
            f"Dossier introuvable : {dossier}\n"
            f"Workspace utilisé : {WORKSPACE_ROOT}"
        )

    if not directory.is_dir():
        return (
            f"Le chemin n'est pas un dossier : "
            f"{dossier}"
        )

    maximum_depth = max(
        1,
        min(profondeur_maximale, 5),
    )

    results = []

    try:
        paths = sorted(
            directory.rglob("*"),
            key=lambda item: str(item).casefold(),
        )

        for path in paths:
            if is_ignored_path(path):
                continue

            relative_to_directory = (
                path.relative_to(directory)
            )

            if (
                len(relative_to_directory.parts)
                > maximum_depth
            ):
                continue

            relative_to_workspace = (
                get_relative_path(path)
            )

            if path.is_dir():
                results.append(
                    f"[DOSSIER] {relative_to_workspace}"
                )

            elif path.is_file():
                file_size = path.stat().st_size

                results.append(
                    f"[FICHIER] {relative_to_workspace} "
                    f"({format_file_size(file_size)})"
                )

            if len(results) >= MAX_LIST_RESULTS:
                results.append(
                    "[LIMITE] Affichage limité à "
                    f"{MAX_LIST_RESULTS} éléments."
                )
                break

    except OSError as error:
        return (
            f"Impossible de parcourir le dossier "
            f"{dossier} : {error}"
        )

    if not results:
        return (
            f"Le dossier '{dossier}' est vide.\n"
            f"Chemin analysé : {directory}"
        )

    return "\n".join(results)


# -------------------------------------------------------------------
# Outil 3 : informations sur un fichier
# -------------------------------------------------------------------

@tool
def informations_fichier(
    chemin: str,
) -> str:
    """
    Retourne les métadonnées d'un fichier du workspace.

    Cet outil ne lit pas le contenu du fichier.
    """

    try:
        path = resolve_workspace_path(
            chemin
        )

    except WorkspaceSecurityError as error:
        return f"Accès refusé : {error}"

    if not path.exists():
        return f"Fichier introuvable : {chemin}"

    if not path.is_file():
        return (
            f"Le chemin n'est pas un fichier : "
            f"{chemin}"
        )

    try:
        statistics = path.stat()

        modified_at = datetime.fromtimestamp(
            statistics.st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

    except OSError as error:
        return (
            f"Impossible de lire les métadonnées "
            f"de {chemin} : {error}"
        )

    extension = (
        path.suffix.lower()
        or "aucune"
    )

    text_supported = (
        path.suffix.lower()
        in ALLOWED_TEXT_EXTENSIONS
    )

    return (
        f"Fichier : {get_relative_path(path)}\n"
        f"Nom : {path.name}\n"
        f"Extension : {extension}\n"
        f"Taille : "
        f"{format_file_size(statistics.st_size)}\n"
        f"Dernière modification : {modified_at}\n"
        f"Lecture texte prise en charge : "
        f"{'oui' if text_supported else 'non'}"
    )


# -------------------------------------------------------------------
# Outil 4 : exploration ciblee
# -------------------------------------------------------------------
@tool
def explorer_workspace_cible(
    termes: list[str],
    dossier: str = ".",
    nombre_maximum_fichiers: int = 20,
) -> str:
    """Classe des fichiers candidats avant lecture detaillee.

    Utilise cet outil avant une analyse multi-fichiers. Lis ensuite seulement
    les meilleurs candidats et produis le livrable sans exploration exhaustive.
    """
    if not isinstance(termes, list):
        return "Erreur : termes doit etre une liste."
    needles = [str(term).strip().casefold() for term in termes if str(term).strip()]
    if not needles:
        return "Erreur : au moins un terme est requis."
    try:
        root = resolve_workspace_path(dossier)
        limit = max(1, min(int(nombre_maximum_fichiers), 40))
    except (WorkspaceSecurityError, TypeError, ValueError) as error:
        return f"Exploration refusee : {error}"
    if not root.is_dir():
        return f"Dossier introuvable : {dossier}"
    scored = []
    for path in iterate_workspace_files(root):
        if path.is_symlink() or path.suffix.lower() not in ALLOWED_TEXT_EXTENSIONS:
            continue
        relative = get_relative_path(path)
        score = sum(4 for term in needles if term in relative.casefold())
        try:
            size = path.stat().st_size
            if score == 0 and size <= 500_000:
                sample = path.read_text(encoding="utf-8", errors="replace")[:12_000].casefold()
                score += sum(min(sample.count(term), 3) for term in needles)
        except OSError:
            continue
        if score:
            scored.append((score, relative, size))
    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    if not scored:
        return (
            "Aucun fichier candidat. Explique le manque ou reponds avec les "
            "informations disponibles ; ne parcours pas tout le workspace."
        )
    lines = [
        f"- score={score} | {relative} | {format_file_size(size)}"
        for score, relative, size in scored[:limit]
    ]
    return (
        "Fichiers candidats classes. Lis uniquement les passages requis, "
        "puis produis le livrable :\n" + "\n".join(lines)
    )


# -------------------------------------------------------------------
# Outil 4 : lire un fichier
# -------------------------------------------------------------------

@tool
def lire_fichier(
    chemin: str,
    debut: int = 0,
    longueur: int = DEFAULT_READ_CHARACTERS,
) -> str:
    """Lit un extrait par caractères ; retourne contenu, pagination et empreinte.

    Utilise lire_extraits_fichiers pour lire plusieurs plages de lignes en un appel.
    """
    try:
        path = resolve_workspace_path(chemin)
        with path_lock(path):
            result = read_workspace_file(chemin, debut, longueur)
            if result.get("ok"):
                import hashlib
                result["content_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.dumps(result, ensure_ascii=False)
    except (OSError, ValueError) as error:
        return json.dumps({"ok": False, "path": chemin, "error": str(error)}, ensure_ascii=False)


# -------------------------------------------------------------------
# Outil 5 : rechercher dans les fichiers
# -------------------------------------------------------------------

@tool
def rechercher_dans_fichiers(
    texte: str,
    dossier: str = ".",
    nombre_maximum_resultats: int = 20,
) -> str:
    """
    Recherche une expression dans les fichiers texte du workspace.

    Retourne :
    - le chemin du fichier ;
    - le numéro de ligne ;
    - un court extrait.

    Utilise cet outil avant de lire entièrement plusieurs fichiers.
    """

    if not texte or not texte.strip():
        return "Le texte recherché ne peut pas être vide."

    try:
        directory = resolve_workspace_path(dossier)

    except WorkspaceSecurityError as error:
        return f"Accès refusé : {error}"

    if not directory.exists():
        return f"Dossier introuvable : {dossier}"

    if not directory.is_dir():
        return f"Le chemin demandé n'est pas un dossier : {dossier}"

    try:
        maximum_results = int(nombre_maximum_resultats)

    except (TypeError, ValueError):
        return (
            "Le nombre maximal de résultats doit être "
            "un nombre entier."
        )

    maximum_results = max(
        1,
        min(maximum_results, MAX_SEARCH_RESULTS),
    )

    searched_text = texte.strip().casefold()
    results = []

    try:
        paths = sorted(directory.rglob("*"))

    except OSError as error:
        return (
            "Impossible de parcourir le dossier "
            f"'{dossier}' : {error}"
        )

    for path in paths:
        if len(results) >= maximum_results:
            break

        if path.is_symlink() or not path.is_file():
            continue

        if is_ignored_path(path):
            continue

        if path.suffix.lower() not in ALLOWED_TEXT_EXTENSIONS:
            continue

        try:
            file_size = path.stat().st_size

            if file_size > MAX_TEXT_FILE_SIZE_BYTES:
                continue

            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as file:
                for line_number, line in enumerate(
                    file,
                    start=1,
                ):
                    if searched_text not in line.casefold():
                        continue

                    clean_line = " ".join(
                        line.strip().split()
                    )

                    excerpt = clean_line[:300]

                    if len(clean_line) > 300:
                        excerpt += "..."

                    results.append(
                        f"- {get_relative_path(path)}:"
                        f"{line_number} | {excerpt}"
                    )

                    if len(results) >= maximum_results:
                        break

        except OSError:
            continue

    if not results:
        return (
            f"Aucune occurrence de '{texte.strip()}' "
            f"trouvée dans le dossier '{dossier}'."
        )

    return (
        f"{len(results)} occurrence(s) trouvée(s) "
        f"pour '{texte.strip()}' :\n"
        + "\n".join(results)
    )


@tool
def executer_commande_controlee(
    commande: list[str],
    dossier: str = ".",
    timeout_secondes: int = 60,
) -> str:
    """Exécute une commande sans shell dans un dossier du workspace.

    Seuls les outils de test et de qualité explicitement autorisés sont
    disponibles. Les arguments sont transmis séparément : aucune expansion
    shell, redirection ou chaîne arbitraire n'est acceptée.
    """
    if not isinstance(commande, list) or not commande or not all(
        isinstance(part, str) and part.strip() for part in commande
    ):
        message = "commande doit être une liste non vide de chaînes."
        record_command(command=commande if isinstance(commande, list) else str(commande), error=message)
        return json.dumps({"ok": False, "error": message})
    executable = Path(commande[0]).name.casefold()
    if executable not in ALLOWED_COMMANDS:
        message = f"Commande refusée. Autorisées : {', '.join(sorted(ALLOWED_COMMANDS))}."
        record_command(command=commande, cwd=dossier, error=message)
        return json.dumps({"ok": False, "error": message})
    started = time.monotonic()
    started_at_ns = time.time_ns()
    session = CURRENT_EXECUTION.get()
    before = {}
    if session:
        import hashlib
        with session._lock:
            previous_paths = list(session.writes)
        for relative in previous_paths:
            path = resolve_workspace_path(relative)
            if path.is_file():
                before[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        working_directory = resolve_workspace_path(dossier)
        if not working_directory.is_dir():
            message = f"Dossier introuvable : {dossier}"
            record_command(command=commande, cwd=dossier, error=message)
            return json.dumps({"ok": False, "error": message})
        timeout = max(1, min(int(timeout_secondes), MAX_COMMAND_TIMEOUT_SECONDS))
        completed = subprocess.run(
            commande,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except (WorkspaceSecurityError, TypeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        message = f"{type(error).__name__}: {error}"
        record_command(
            command=commande,
            cwd=dossier,
            error=message,
            duration_seconds=time.monotonic() - started,
        )
        return json.dumps({"ok": False, "error": message}, ensure_ascii=False)
    stdout = (completed.stdout or "")[-MAX_COMMAND_OUTPUT_CHARACTERS:]
    stderr = (completed.stderr or "")[-MAX_COMMAND_OUTPUT_CHARACTERS:]
    record_command(
        command=commande,
        cwd=get_relative_path(working_directory),
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=time.monotonic() - started,
    )
    session = CURRENT_EXECUTION.get()
    checked_files = {}
    if session:
        import hashlib
        with session._lock:
            written_paths = list(session.writes)
        for relative in written_paths:
            path = resolve_workspace_path(relative)
            if path.is_file():
                checked_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.dumps({
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "category": command_category(commande),
        "command": commande,
        "checked_files": checked_files,
        "started_at_ns": started_at_ns,
        "completed_at_ns": time.time_ns(),
        "stable_during_check": before == checked_files,
        "cwd": get_relative_path(working_directory),
        "stdout": stdout,
        "stderr": stderr,
    }, ensure_ascii=False)


# -------------------------------------------------------------------
# Liste des outils exportés
# -------------------------------------------------------------------

@tool
def lire_extraits_fichiers(extraits: list[dict[str, str | int]]) -> str:
    """Lit plusieurs extraits en un appel : [{chemin, debut_ligne?, fin_ligne?}].

    Retourne chemins, numéros de ligne et empreintes ; budget total 40000 caractères.
    """
    results = []
    remaining = MAX_READ_CHARACTERS
    for request in extraits[:12]:
        if remaining <= 0:
            break
        name = str(request.get("chemin", ""))
        try:
            path = resolve_workspace_path(name)
            validate_text_file(path)
            if path.stat().st_size > MAX_TEXT_FILE_SIZE_BYTES:
                raise ValueError("Fichier trop volumineux")
            start = max(1, int(request.get("debut_ligne", 1)))
            end = max(start, int(request.get("fin_ligne", start + 199)))
            with path_lock(path):
                text = path.read_text(encoding="utf-8", errors="replace")
                session = CURRENT_EXECUTION.get()
                if session is not None:
                    session.observe(path)
            import hashlib
            lines = text.splitlines()
            selected = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines) if start <= i + 1 <= end)
            clipped = selected[:remaining]
            remaining -= len(clipped)
            results.append({"path": name, "ok": True, "content": clipped,
                            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                            "truncated": len(clipped) < len(selected), "total_lines": len(lines)})
        except (OSError, ValueError, TypeError) as error:
            results.append({"path": name, "ok": False, "error": str(error)})
    return json.dumps({"files": results, "requests_omitted": max(0, len(extraits) - len(results))}, ensure_ascii=False)


@tool
def consulter_diff(chemin: str = ".") -> str:
    """Affiche les différences Git du workspace (sortie limitée, sans modification)."""
    path = resolve_workspace_path(chemin)
    session = CURRENT_EXECUTION.get()
    if session is not None:
        from tools.write_tools import _get_proposal
        changes = []
        for relative, record in session.writes.items():
            if chemin != "." and resolve_workspace_path(relative) != path:
                continue
            try:
                proposal = _get_proposal(record["proposal_id"])
                changes.append({"path": relative, "diff": proposal.get("diff", "")})
            except (KeyError, RuntimeError):
                continue
        if changes:
            text = json.dumps({"ok": True, "changes": changes}, ensure_ascii=False)
            if len(text) <= MAX_COMMAND_OUTPUT_CHARACTERS:
                return text
            return json.dumps({"ok": True, "diff": "\n".join(item["diff"] for item in changes)[:MAX_COMMAND_OUTPUT_CHARACTERS],
                               "truncated": True}, ensure_ascii=False)
    completed = subprocess.run(["git", "diff", "--no-ext-diff", "--no-textconv", "--", str(path)],
                               cwd=WORKSPACE_ROOT, capture_output=True, text=True, timeout=30, check=False)
    return json.dumps({"ok": completed.returncode == 0, "returncode": completed.returncode,
                       "diff": completed.stdout[:MAX_COMMAND_OUTPUT_CHARACTERS],
                       "truncated": len(completed.stdout) > MAX_COMMAND_OUTPUT_CHARACTERS,
                       "error": completed.stderr[:1000]}, ensure_ascii=False)


@tool
def lire_resultat_tache(identifiant: str, debut: int = 0, longueur: int = 8000) -> str:
    """Lit à la demande le livrable complet d'une tâche dépendante, sans le régénérer."""
    session = CURRENT_EXECUTION.get()
    result = session.artifacts.get(identifiant) if session else None
    if result is None:
        return json.dumps({"ok": False, "error": "Livrable indisponible ou hors dépendances."})
    start = max(0, debut)
    end = start + max(1, min(longueur, MAX_READ_CHARACTERS))
    return json.dumps({"ok": True, "task": identifiant, "content": result[start:end],
                       "has_more": end < len(result), "next": min(end, len(result))}, ensure_ascii=False)


# Execution tools are deliberately separate: a reader cannot run arbitrary
# Python/Node or Git commands, which can themselves modify files.
READ_ONLY_TOOLS = [
    informations_workspace,
    lister_fichiers,
    informations_fichier,
    explorer_workspace_cible,
    lire_fichier,
    rechercher_dans_fichiers,
    lire_extraits_fichiers,
    consulter_diff,
    lire_resultat_tache,
]

EXECUTION_TOOLS = [executer_commande_controlee]
