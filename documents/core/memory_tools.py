"""
Mémoire compacte de la conversation pour l'agent.

Problème résolu : l'API Bedrock est sans état. Renvoyer tout l'historique
brut à chaque tour coûte cher en tokens. On combine donc :

1. une fenêtre glissante : seuls les derniers messages bruts sont renvoyés ;
2. une mémoire compacte et bornée, injectée en tête du prompt système,
   contenant le contexte durable (résumé, suivi de tâches, décisions, fichiers).

La mémoire est mise à jour par l'agent lui-même via l'outil
`mettre_a_jour_memoire` (approche « memory tool »). Cela évite un appel
LLM dédié : la mémoire est écrite dans le même tour, quasi sans surcoût.

Ce module ne dépend ni de Streamlit ni du stockage : il expose des
fonctions pures et un outil LangChain. L'orchestration (persistance,
session) est faite dans app.py.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool


# -------------------------------------------------------------------
# Bornes de taille (garde-fous anti-explosion de tokens)
# -------------------------------------------------------------------

# Nombre de messages bruts (user/assistant) renvoyés à l'API.
# En deçà, l'historique est complété par la mémoire compacte.
SLIDING_WINDOW_MESSAGES = 12

# Bornes de la mémoire compacte.
MAX_SUMMARY_CHARACTERS = 1200
MAX_ITEMS = 15
MAX_ITEM_CHARACTERS = 160

# Statuts autorisés pour une tâche.
TASK_STATUSES = ("à faire", "en cours", "terminé")


# -------------------------------------------------------------------
# Structure de mémoire
# -------------------------------------------------------------------

def empty_memory() -> dict[str, Any]:
    """Retourne une mémoire vide."""

    return {
        "summary": "",
        "tasks": [],
        "decisions": [],
        "files": [],
    }


def _truncate(text: str, limit: int) -> str:
    """Tronque proprement une chaîne à une longueur maximale."""

    text = " ".join(str(text).split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "…"


def _normalize_tasks(
    tasks: list[Any] | None,
) -> list[dict[str, str]]:
    """
    Normalise la liste des tâches en entrées {label, status}.

    Accepte des chaînes (statut par défaut « à faire ») ou des
    dictionnaires {label/tache, status/statut}.
    """

    normalized: list[dict[str, str]] = []

    for item in (tasks or [])[:MAX_ITEMS]:
        if isinstance(item, str):
            label = item
            status = "à faire"

        elif isinstance(item, dict):
            label = item.get("label") or item.get("tache") or ""
            status = (
                item.get("status")
                or item.get("statut")
                or "à faire"
            )

        else:
            continue

        label = _truncate(label, MAX_ITEM_CHARACTERS)

        if not label:
            continue

        if status not in TASK_STATUSES:
            status = "à faire"

        normalized.append(
            {"label": label, "status": status}
        )

    return normalized


def _normalize_list(items: list[Any] | None) -> list[str]:
    """Normalise une liste de chaînes courtes, sans doublon."""

    normalized: list[str] = []

    for item in (items or [])[:MAX_ITEMS]:
        text = _truncate(item, MAX_ITEM_CHARACTERS)

        if text and text not in normalized:
            normalized.append(text)

    return normalized


def normalize_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    """
    Nettoie et borne une structure de mémoire quelconque.
    """

    memory = memory or {}

    return {
        "summary": _truncate(
            memory.get("summary", ""),
            MAX_SUMMARY_CHARACTERS,
        ),
        "tasks": _normalize_tasks(memory.get("tasks")),
        "decisions": _normalize_list(memory.get("decisions")),
        "files": _normalize_list(memory.get("files")),
    }


def memory_is_empty(memory: dict[str, Any] | None) -> bool:
    """Indique si une mémoire ne contient aucune information."""

    memory = normalize_memory(memory)

    return not (
        memory["summary"]
        or memory["tasks"]
        or memory["decisions"]
        or memory["files"]
    )


# -------------------------------------------------------------------
# Rendu texte injecté dans le prompt système
# -------------------------------------------------------------------

_STATUS_MARKERS = {
    "à faire": "[à faire]",
    "en cours": "[en cours]",
    "terminé": "[terminé]",
}


def format_memory_block(memory: dict[str, Any] | None) -> str:
    """
    Rend la mémoire sous forme d'un bloc texte compact et structuré,
    destiné à être injecté dans le message système.

    Retourne une chaîne vide si la mémoire est vide.
    """

    memory = normalize_memory(memory)

    if memory_is_empty(memory):
        return ""

    lines: list[str] = ["[MÉMOIRE DE LA CONVERSATION]"]

    if memory["summary"]:
        lines.append(f"Résumé : {memory['summary']}")

    if memory["tasks"]:
        lines.append("Tâches :")

        for task in memory["tasks"]:
            marker = _STATUS_MARKERS.get(
                task["status"],
                "[à faire]",
            )
            lines.append(f"- {marker} {task['label']}")

    if memory["decisions"]:
        lines.append(
            "Décisions : " + " ; ".join(memory["decisions"])
        )

    if memory["files"]:
        lines.append(
            "Fichiers concernés : "
            + ", ".join(memory["files"])
        )

    return "\n".join(lines)


# -------------------------------------------------------------------
# Fenêtre glissante
# -------------------------------------------------------------------

def apply_sliding_window(
    messages: list[dict[str, Any]],
    window: int = SLIDING_WINDOW_MESSAGES,
) -> list[dict[str, Any]]:
    """
    Ne conserve que les derniers messages pour l'appel à l'API.

    La mémoire compacte compense la perte des messages plus anciens.
    """

    if window <= 0 or len(messages) <= window:
        return list(messages)

    return list(messages[-window:])


# -------------------------------------------------------------------
# Outil exposé à l'agent
# -------------------------------------------------------------------

# La mémoire courante est fournie par app.py via ce conteneur mutable.
# L'outil lit/écrit ici ; app.py lit le résultat après agent.invoke.
_MEMORY_STATE: dict[str, Any] = {"memory": empty_memory()}


def get_memory_state() -> dict[str, Any]:
    """Retourne la mémoire courante (normalisée)."""

    return normalize_memory(_MEMORY_STATE["memory"])


def set_memory_state(memory: dict[str, Any] | None) -> None:
    """Injecte la mémoire courante avant un appel à l'agent."""

    _MEMORY_STATE["memory"] = normalize_memory(memory)


@tool
def mettre_a_jour_memoire(
    summary: str | None = None,
    tasks: list[dict[str, str]] | None = None,
    decisions: list[str] | None = None,
    files: list[str] | None = None,
) -> str:
    """
    Met à jour la mémoire durable de la conversation.

    Appelle cet outil UNIQUEMENT quand un fait important doit être
    retenu au-delà des prochains messages : avancement d'une tâche,
    décision prise, contexte structurant, fichier central du travail
    en cours. N'utilise pas cet outil pour du bavardage.

    Cette mémoire est réinjectée automatiquement au début de chaque
    futur échange : elle te sert de « souvenir » compact et économe
    en tokens, même après un redémarrage de l'application.

    Paramètres :
    - summary : résumé global concis du contexte (remplace l'ancien).
    - tasks : liste de tâches, chacune {"label": str, "status": str}
      avec status parmi "à faire", "en cours", "terminé".
    - decisions : liste de décisions importantes actées.
    - files : liste des fichiers centraux du travail en cours.

    Ne fournis que les champs à mettre à jour ; les champs omis
    conservent leur valeur précédente.
    """

    current = normalize_memory(_MEMORY_STATE["memory"])

    updated = {
        "summary": (
            summary
            if summary is not None
            else current["summary"]
        ),
        "tasks": (
            tasks
            if tasks is not None
            else current["tasks"]
        ),
        "decisions": (
            decisions
            if decisions is not None
            else current["decisions"]
        ),
        "files": (
            files
            if files is not None
            else current["files"]
        ),
    }

    _MEMORY_STATE["memory"] = normalize_memory(updated)

    return (
        "Mémoire mise à jour. Elle sera réinjectée aux prochains "
        "échanges."
    )


MEMORY_TOOLS = [mettre_a_jour_memoire]
