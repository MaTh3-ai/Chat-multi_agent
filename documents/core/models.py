"""Catalogue central des modèles Amazon Bedrock autorisés."""

from copy import deepcopy
from typing import Any


DEFAULT_MODEL_ALIAS = "claude-sonnet-4.5"
DEFAULT_REGION = "eu-west-3"


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "claude-sonnet-4.5": {
        "label": "Claude Sonnet 4.5",
        "mode": "Équilibré",
        "description": (
            "Modèle polyvalent pour les conversations, "
            "le code et les usages courants."
        ),
        "model_id": (
            "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
        ),
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": 0.2,
        "supports_images": True,
        "supports_tools": True,
        "enabled": True,
    },
    "claude-opus-4.7": {
        "label": "Claude Opus 4.7",
        "mode": "Approfondi",
        "description": (
            "Modèle avancé pour les tâches et raisonnements complexes."
        ),
        "model_id": "eu.anthropic.claude-opus-4-7",
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": None,
        "supports_images": True,
        "supports_tools": True,
        "enabled": True,
    },
    "claude-opus-4.8": {
        "label": "Claude Opus 4.8",
        "mode": "Expert",
        "description": (
            "Modèle haut de gamme pour l'analyse, "
            "la critique et la synthèse."
        ),
        "model_id": "eu.anthropic.claude-opus-4-8",
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": None,
        "supports_images": True,
        "supports_tools": True,
        "enabled": True,
    },
    "claude-haiku-4.5": {
        "label": "Claude Haiku 4.5",
        "mode": "Rapide",
        "description": (
            "Modèle économique et multimodal, idéal pour "
            "l'exécution, la rédaction et les résumés."
        ),
        "model_id": (
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        ),
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": 0.2,
        "supports_images": True,
        "supports_tools": True,
        "enabled": True,
    },
    "claude-3-haiku": {
        "label": "Claude 3 Haiku",
        "mode": "Rapide",
        "description": (
            "Ancien modèle Anthropic très bon marché, "
            "utile pour les tâches légères."
        ),
        "model_id": (
            "eu.anthropic.claude-3-haiku-20240307-v1:0"
        ),
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 4096,
        "temperature": 0.2,
        "supports_images": True,
        "supports_tools": True,
        "enabled": True,
    },
    "nova-lite": {
        "label": "Nova Lite",
        "mode": "Rapide",
        "description": (
            "Modèle Amazon économique et multimodal "
            "pour les tâches courantes."
        ),
        "model_id": "eu.amazon.nova-lite-v1:0",
        "region": DEFAULT_REGION,
        "provider": "amazon",
        "max_tokens": 4096,
        "temperature": 0.2,
        "supports_images": True,
        "supports_tools": True,
        "enabled": True,
    },
    "nova-micro": {
        "label": "Nova Micro",
        "mode": "Nano",
        "description": (
            "Le modèle Amazon le moins cher, texte uniquement, "
            "pour le tri et le routing."
        ),
        "model_id": "eu.amazon.nova-micro-v1:0",
        "region": DEFAULT_REGION,
        "provider": "amazon",
        "max_tokens": 4096,
        "temperature": 0.2,
        "supports_images": False,
        "supports_tools": True,
        "enabled": True,
    },
}


def get_model_config(model_alias: str) -> dict[str, Any]:
    """Retourne une copie vérifiée de la configuration d'un modèle."""

    if model_alias not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))

        raise ValueError(
            f"Modèle inconnu : {model_alias}. "
            f"Modèles déclarés : {available}"
        )

    model_config = deepcopy(MODEL_REGISTRY[model_alias])

    if not model_config.get("enabled", False):
        raise ValueError(
            f"Le modèle {model_config['label']} est désactivé."
        )

    model_id = model_config.get("model_id", "")

    if not model_id:
        raise ValueError(
            f"Le modèle {model_config['label']} ne possède pas d'ID."
        )

    if not model_id.startswith("eu."):
        raise ValueError(
            "Modèle refusé : seuls les profils européens "
            "commençant par 'eu.' sont autorisés."
        )

    return model_config


def get_available_models() -> dict[str, str]:
    """Retourne les modèles disponibles pour l'interface."""

    return {
        alias: config["label"]
        for alias, config in MODEL_REGISTRY.items()
        if config.get("enabled", False)
    }


def get_model_options() -> dict[str, str]:
    """Retourne les libellés détaillés destinés à l'interface."""

    return {
        alias: f"{config['mode']} - {config['label']}"
        for alias, config in MODEL_REGISTRY.items()
        if config.get("enabled", False)
    }


def get_default_model_alias() -> str:
    """Retourne l'alias du modèle par défaut après validation."""

    get_model_config(DEFAULT_MODEL_ALIAS)
    return DEFAULT_MODEL_ALIAS


# ─────────────────────────────────────────────────────────────────────────────
# Capacités multimodales
# ─────────────────────────────────────────────────────────────────────────────

# Formats d'images supportés par Bedrock Converse (Anthropic)
BEDROCK_IMAGE_FORMATS = {"png", "jpeg", "jpg", "gif", "webp"}

# Formats de documents supportés par Bedrock Converse (Anthropic)
BEDROCK_DOCUMENT_FORMATS = {
    "pdf", "csv", "doc", "docx", "xls", "xlsx",
    "html", "txt", "md"
}


def get_model_capabilities(model_alias: str) -> dict[str, Any]:
    """
    Retourne les capacités multimodales d'un modèle.
    
    Retourne un dict avec :
    - supports_images (bool)
    - supports_documents (bool)
    - accepted_image_formats (set)
    - accepted_document_formats (set)
    """
    config = get_model_config(model_alias)
    supports_images = config.get("supports_images", False)
    
    # Pour l'instant, tous les modèles Anthropic via Bedrock Converse
    # qui supportent les images supportent aussi les documents
    supports_documents = supports_images
    
    return {
        "supports_images": supports_images,
        "supports_documents": supports_documents,
        "accepted_image_formats": BEDROCK_IMAGE_FORMATS if supports_images else set(),
        "accepted_document_formats": BEDROCK_DOCUMENT_FORMATS if supports_documents else set(),
    }


def classify_attachment(filename: str, capabilities: dict[str, Any]) -> str | None:
    """
    Classifie un fichier joint selon son extension.
    
    Retourne :
    - "image" si c'est une image supportée
    - "document" si c'est un document supporté
    - None si le format n'est pas supporté
    """
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    
    if ext in capabilities["accepted_image_formats"]:
        return "image"
    elif ext in capabilities["accepted_document_formats"]:
        return "document"
    else:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tiers de modèles (système multi-agent)
# ─────────────────────────────────────────────────────────────────────────────
#
# Menu FERMÉ de profils. Le Chef de projet ne choisit QUE parmi ces tiers,
# jamais un model_id libre. Cela garde le coût sous contrôle (best practice
# "tiered model routing"). Chaque tier pointe vers un alias du MODEL_REGISTRY.

MODEL_TIERS: dict[str, dict[str, Any]] = {
    "nano": {
        "label": "Nano (tri / routing)",
        "model_alias": "nova-micro",
        "description": (
            "Le moins cher. Classification, routing, tâches triviales. "
            "Texte uniquement."
        ),
        "relative_cost": 1,
    },
    "fast": {
        "label": "Rapide (exécution)",
        "model_alias": "claude-haiku-4.5",
        "description": (
            "Bon marché et multimodal. Exécution, rédaction, résumés. "
            "Cheval de bataille des workers."
        ),
        "relative_cost": 4,
    },
    "smart": {
        "label": "Sérieux (code / analyse)",
        "model_alias": "claude-sonnet-4.5",
        "description": (
            "Travail sérieux : code, analyse, raisonnement structuré."
        ),
        "relative_cost": 15,
    },
    "expert": {
        "label": "Expert (chef / arbitrage)",
        "model_alias": "claude-opus-4.8",
        "description": (
            "Réservé au Chef de projet et aux arbitrages critiques rares."
        ),
        "relative_cost": 60,
    },
}

# Tier par défaut d'un worker si le Chef n'en précise pas.
DEFAULT_WORKER_TIER = "fast"

# Tier imposé au Chef de projet.
CHIEF_TIER = "expert"


def get_tier(tier_name: str) -> dict[str, Any]:
    """Retourne la config d'un tier vérifié (menu fermé)."""

    if tier_name not in MODEL_TIERS:
        available = ", ".join(sorted(MODEL_TIERS))
        raise ValueError(
            f"Tier inconnu : {tier_name}. Tiers autorisés : {available}"
        )
    return deepcopy(MODEL_TIERS[tier_name])


def resolve_tier_to_model_alias(tier_name: str) -> str:
    """Traduit un nom de tier en alias de modèle valide du registre."""

    tier = get_tier(tier_name)
    alias = tier["model_alias"]
    # Vérifie que le modèle ciblé existe et est activé.
    get_model_config(alias)
    return alias


def get_tier_options() -> dict[str, str]:
    """Retourne les tiers destinés à l'interface (label lisible)."""

    return {name: cfg["label"] for name, cfg in MODEL_TIERS.items()}


def model_supports_temperature(model_alias: str) -> bool:
    """Indique si un modèle accepte le paramètre `temperature`.

    Certains modèles récents (ex. Claude Opus 4.7 / 4.8) rejettent
    `temperature` côté Bedrock Converse ('temperature is deprecated for
    this model'). Ces modèles sont déclarés avec `temperature: None` dans
    le registre : on s'appuie sur cette convention comme source de vérité.
    """

    config = get_model_config(model_alias)
    return config.get("temperature") is not None


def sanitize_temperature(model_alias: str, temperature):
    """Retourne une température sûre pour le modèle, ou None.

    - Si le modèle n'accepte pas `temperature`, renvoie None (le paramètre
      ne sera pas transmis à l'API).
    - Sinon, renvoie la température demandée, ou la valeur par défaut du
      registre si `temperature` vaut None.
    """

    if not model_supports_temperature(model_alias):
        return None
    if temperature is not None:
        return temperature
    return get_model_config(model_alias).get("temperature")

# ─────────────────────────────────────────────────────────────────────────────
# Profils de prompting (réécriture légère des demandes)
# ─────────────────────────────────────────────────────────────────────────────
#
# Chaque modèle destinataire est associé à un profil de prompting utilisé par
# le "modèle prompteur" (core/prompt_rewriter.py) pour adapter la FORME du
# prompt sans en modifier le fond :
#   - light    : petits modèles (Nano, Haiku, Nova) -> objectif très explicite
#   - balanced : modèles intermédiaires (Sonnet)     -> objectif + critères
#   - frontier : modèles puissants (Opus, raisonnement) -> résultat + liberté
#
# Cette table est le SEUL endroit à modifier pour changer le mapping.

PROMPTING_PROFILE_BY_ALIAS: dict[str, str] = {
    "nova-micro": "light",
    "nova-lite": "light",
    "claude-3-haiku": "light",
    "claude-haiku-4.5": "light",
    "claude-sonnet-4.5": "balanced",
    "claude-opus-4.7": "frontier",
    "claude-opus-4.8": "frontier",
}

# Profil retenu si l'alias n'est pas répertorié ci-dessus.
DEFAULT_PROMPTING_PROFILE = "balanced"

# Modèle utilisé PAR DÉFAUT comme prompteur (léger, rapide, multimodal).
# Option B validée : Claude Haiku 4.5.
DEFAULT_PROMPTER_MODEL_ALIAS = "claude-haiku-4.5"


def get_prompting_profile(model_alias: str) -> str:
    """Retourne le profil de prompting (light|balanced|frontier) d'un alias."""

    return PROMPTING_PROFILE_BY_ALIAS.get(
        model_alias, DEFAULT_PROMPTING_PROFILE
    )

