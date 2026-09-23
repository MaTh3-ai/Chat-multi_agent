"""Modèle prompteur : réécriture légère des demandes utilisateur.

Ce module prépare (reformule) la demande d'un utilisateur AVANT de la
transmettre au modèle principal ou à un agent. Il :

- n'accède jamais au workspace ni au code de l'application ;
- n'appelle aucun outil ;
- ne résout jamais la demande lui-même ;
- reste une optimisation NON bloquante : toute erreur -> message original.

Point d'entrée public : ``rewrite_user_prompt(...)`` (ajouté plus bas).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# La suite du module (constantes, RewriteResult, prompt système, contexte,
# appel LLM, parsing JSON, fallback et rewrite_user_prompt) est ajoutée par
# insertions ciblées successives après ce marqueur.
# --- REWRITER_BUILD_MARKER ---

import json
import logging

from core.models import (
    DEFAULT_PROMPTER_MODEL_ALIAS,
    get_prompting_profile,
)

logger = logging.getLogger(__name__)

# Longueur maximale de la demande et du prompt reformule. Cette borne
# technique ne desactive jamais la reformulation : elle evite seulement un
# appel demesure. Le controle fonctionnel reste le toggle Streamlit.
MAX_REWRITE_CHARS = 20_000

# Budget suffisant pour reformuler fidelement une demande detaillee.
PROMPTER_MAX_TOKENS = 4096

# Température basse : reformulation fidèle, peu créative.
PROMPTER_TEMPERATURE = 0.1

# --- Mapping mode -> consigne de reformulation -----------------------------

# Le "mode" reflète l'intention d'usage du prompt reformulé.
MODE_INSTRUCTIONS = {
    "chat": (
        "Reformule la demande pour un assistant conversationnel généraliste. "
        "Clarifie l'intention sans ajouter d'hypothèses non exprimées."
    ),
    "agent": (
        "Reformule la demande pour un agent autonome qui devra planifier et "
        "exécuter des étapes. Rends l'objectif et les livrables explicites."
    ),
}

DEFAULT_MODE = "chat"


@dataclass
class RewriteResult:
    """Résultat d'une réécriture de prompt.

    ``ok`` indique si la réécriture a réussi. En cas d'échec, ``text`` contient
    le message original (fallback non bloquant) et ``error`` la cause.
    """

    text: str
    ok: bool = True
    profile: str = ""
    model_alias: str = ""
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def fallback(cls, original: str, error: str) -> "RewriteResult":
        """Construit un résultat de repli conservant le message original."""
        return cls(text=original, ok=False, error=error)


# --- Prompt système du prompteur -------------------------------------------

def build_system_prompt(profile: str, mode: str) -> str:
    """Construit le prompt système du modèle prompteur.

    Le comportement dépend du profil (``light`` / ``balanced`` / ``frontier``)
    et du mode d'usage cible.
    """
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS[DEFAULT_MODE])

    base = (
        "Tu es un module de reformulation de requêtes. Ton unique rôle est de "
        "réécrire la demande de l'utilisateur pour la rendre plus claire, "
        "structurée et exploitable. Tu NE réponds jamais à la demande, tu ne "
        "résous rien, tu n'inventes aucune information et tu ne demandes "
        "jamais de précision. "
        + mode_instruction
    )

    if profile == "light":
        depth = (
            " Interviens au minimum : corrige seulement la formulation si elle "
            "est ambiguë, sinon conserve la demande quasi intacte."
            "Évite les formulations de type « Peux-tu ? » ou « Pourrais-tu ? » ; formule systématiquement des consignes directes, claires et impératives."
        )
    elif profile == "frontier":
        depth = (
            " Structure en profondeur : dégage l'objectif, les contraintes, les "
            "livrables attendus et les critères de réussite, sans jamais "
            "extrapoler au-delà de ce qui est exprimé."
            "Évite les formulations de type « Peux-tu ? » ou « Pourrais-tu ? » ; formule systématiquement des consignes directes, claires et impératives."
        )
    else:  # balanced
        depth = (
            " Clarifie l'objectif et les contraintes principales de façon "
            "concise, en restant fidèle à l'intention d'origine."
            "Évite les formulations de type « Peux-tu ? » ou « Pourrais-tu ? » ; formule systématiquement des consignes directes, claires et impératives."
        )

    fmt = (
        " Réponds STRICTEMENT en JSON valide, sans texte hors du JSON, au "
        'format : {"rewritten": "<prompt reformulé>", "notes": ["<note>", ...]}'
    )

    return base + depth + fmt


# --- Construction du contexte court ----------------------------------------

def build_user_payload(original: str, context: str | None = None) -> str:
    """Assemble le message transmis au prompteur.

    Le contexte est volontairement court et facultatif. La demande d'origine est transmise jusqu'a ``MAX_REWRITE_CHARS``.
    La reformulation n'est jamais desactivee selon le nombre de lignes ou
    le niveau de detail : seul le toggle Streamlit decide de l'activer.
    """
    original = (original or "").strip()
    if len(original) > MAX_REWRITE_CHARS:
        original = original[:MAX_REWRITE_CHARS]

    parts = ["DEMANDE ORIGINALE :", original]

    if context:
        context = context.strip()
        if len(context) > 1000:
            context = context[:1000]
        parts.extend(["", "CONTEXTE (indicatif, ne pas y répondre) :", context])

    return "\n".join(parts)

# --- Appel LLM + parsing JSON robuste --------------------------------------

def _extract_json_object(raw: str) -> dict:
    """Extrait un objet JSON d'une réponse LLM, même entourée de texte.

    Tente d'abord un parsing direct, puis se rabat sur la première accolade
    ouvrante et la dernière fermante. Lève ``ValueError`` si rien d'exploitable.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("réponse vide du prompteur")

    # Retrait d'éventuelles clôtures Markdown ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("`")
        # après strip des backticks, un préfixe "json" peut subsister
        if raw[:4].lower() == "json":
            raw = raw[4:]
        raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("aucun objet JSON exploitable dans la réponse du prompteur")


def _call_prompter_llm(system_prompt: str, user_payload: str, model_alias: str) -> str:
    """Appelle le modèle prompteur et renvoie le texte brut de la réponse."""
    from llm_engine import create_chat_model

    model = create_chat_model(
        model_alias=model_alias,
        temperature=PROMPTER_TEMPERATURE,
        max_tokens=PROMPTER_MAX_TOKENS,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    response = model.invoke(messages)
    # LangChain renvoie un objet message ; on récupère .content, sinon str().
    content = getattr(response, "content", None)
    if content is None:
        content = str(response)
    if isinstance(content, list):
        # certains providers renvoient une liste de blocs {type, text}
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        content = "".join(parts)
    return str(content)


# --- Point d'entrée public --------------------------------------------------

def rewrite_user_prompt(
    original: str,
    *,
    mode: str = DEFAULT_MODE,
    context: str | None = None,
    model_alias: str | None = None,
) -> RewriteResult:
    """Réécrit une demande utilisateur de façon NON bloquante.

    En cas d'erreur (profil inconnu, appel LLM en échec, JSON invalide, etc.),
    renvoie un ``RewriteResult`` de repli contenant le message original.
    """
    original = (original or "").strip()
    if not original:
        return RewriteResult.fallback("", "demande vide")

    alias = model_alias or DEFAULT_PROMPTER_MODEL_ALIAS

    try:
        # Le profil (light|balanced|frontier) dépend du MODÈLE prompteur réel,
        # pas du mode d'usage. On résout donc l'alias d'abord.
        profile = get_prompting_profile(alias)
    except Exception as exc:  # profil indisponible -> repli
        logger.warning("Profil de prompting indisponible : %s", exc)
        return RewriteResult.fallback(original, f"profil indisponible : {exc}")

    system_prompt = build_system_prompt(profile, mode)
    user_payload = build_user_payload(original, context=context)

    try:
        raw = _call_prompter_llm(system_prompt, user_payload, alias)
    except Exception as exc:  # échec réseau/modèle -> repli non bloquant
        logger.warning("Appel prompteur en échec : %s", exc)
        return RewriteResult.fallback(original, f"appel LLM en échec : {exc}")

    try:
        obj = _extract_json_object(raw)
    except Exception as exc:  # JSON illisible -> repli non bloquant
        logger.warning("Réponse prompteur non parsable : %s", exc)
        return RewriteResult.fallback(original, f"réponse non parsable : {exc}")

    rewritten = str(obj.get("rewritten", "")).strip()
    if not rewritten:
        return RewriteResult.fallback(original, "champ 'rewritten' vide")

    if len(rewritten) > MAX_REWRITE_CHARS:
        rewritten = rewritten[:MAX_REWRITE_CHARS]

    notes_raw = obj.get("notes", [])
    notes = [str(n).strip() for n in notes_raw if str(n).strip()] if isinstance(notes_raw, list) else []

    return RewriteResult(
        text=rewritten,
        ok=True,
        profile=profile,
        model_alias=alias,
        notes=notes,
    )

