"""Catalogue central des modèles Amazon Bedrock autorisés."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_MODEL_ALIAS = "claude-sonnet-5.5"
DEFAULT_REGION = "eu-west-3"


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "nova-2-lite": {
        "label": "Amazon Nova 2 Lite",
        "mode": "Nano",
        "description": (
            "Modèle Amazon économique et multimodal pour le routage, "
            "la classification et l'extraction structurée."
        ),
        "model_id": "eu.amazon.nova-2-lite-v1:0",
        "region": DEFAULT_REGION,
        "provider": "amazon",
        "max_tokens": 8_192,
        "temperature": 0.2,
        "supports_images": True,
        "supports_documents": False,
        "supports_tools": True,
        "enabled": True,
    },
    "claude-haiku-4.5": {
        "label": "Claude Haiku 4.5",
        "mode": "Rapide",
        "description": (
            "Modèle économique pour les agents spécialisés, l'exécution ciblée, "
            "la rédaction courte et les résumés."
        ),
        "model_id": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 8_192,
        "temperature": 0.2,
        "supports_images": True,
        "supports_documents": True,
        "supports_tools": True,
        "enabled": True,
    },
    "claude-sonnet-5.5": {
        "label": "Claude Sonnet 5.5",
        "mode": "Sérieux",
        "description": (
            "Modèle principal pour le code, l'analyse, les statistiques, "
            "la vision et les tâches complexes."
        ),
        "model_id": "eu.anthropic.claude-sonnet-5-5",
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 16_384,
        "temperature": None,
        "supports_images": True,
        "supports_documents": True,
        "supports_tools": True,
        "enabled": True,
    },
    "claude-opus-5.5": {
        "label": "Claude Opus 5.5",
        "mode": "Expert",
        "description": (
            "Modèle de haut niveau pour la direction du système, l'arbitrage, "
            "la critique et la synthèse finale."
        ),
        "model_id": "eu.anthropic.claude-opus-5-5",
        "region": DEFAULT_REGION,
        "provider": "anthropic",
        "max_tokens": 16_384,
        "temperature": None,
        "supports_images": True,
        "supports_documents": True,
        "supports_tools": True,
        "enabled": True,
    },
    "pixtral-large": {
        "label": "Mistral Pixtral Large 25.02",
        "mode": "Vision",
        "description": (
            "Modèle multimodal spécialisé pour l'analyse indépendante "
            "d'images, de graphiques et de documents visuels."
        ),
        "model_id": "eu.mistral.pixtral-large-2502-v1:0",
        "region": DEFAULT_REGION,
        "provider": "mistral",
        "max_tokens": 16_384,
        "temperature": 0.2,
        "supports_images": True,
        "supports_documents": False,
        "supports_tools": True,
        "enabled": True,
    },
}


BEDROCK_IMAGE_FORMATS = frozenset(
    {
        "png",
        "jpeg",
        "jpg",
        "gif",
        "webp",
    }
)

BEDROCK_DOCUMENT_FORMATS = frozenset(
    {
        "pdf",
        "csv",
        "doc",
        "docx",
        "xls",
        "xlsx",
        "html",
        "txt",
        "md",
    }
)


def get_model_config(model_alias: str) -> dict[str, Any]:
    """Retourne une copie valide de la configuration d'un modèle."""
    if model_alias not in MODEL_REGISTRY:
        declared_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Modèle inconnu : {model_alias}. "
            f"Modèles déclarés : {declared_models}"
        )

    config = deepcopy(MODEL_REGISTRY[model_alias])

    if not config.get("enabled", False):
        raise ValueError(f"Le modèle {config['label']} est désactivé.")

    model_id = str(config.get("model_id", "")).strip()

    if not model_id:
        raise ValueError(
            f"Le modèle {config['label']} ne possède pas d'identifiant."
        )

    if not model_id.startswith("eu."):
        raise ValueError(
            "Seuls les profils d'inférence européens commençant par "
            "'eu.' sont autorisés."
        )

    return config


def sanitize_temperature(
    model_alias: str,
    temperature: float | int | None,
) -> float | None:
    """Retourne une température valide ou None si elle ne doit pas être envoyée."""
    config = get_model_config(model_alias)
    default_temperature = config.get("temperature")

    # Les modèles concernés ne doivent recevoir aucun paramètre temperature.
    if default_temperature is None:
        return None

    if temperature is None:
        temperature = default_temperature

    if isinstance(temperature, bool) or not isinstance(
        temperature,
        (int, float),
    ):
        raise ValueError("La température doit être un nombre ou None.")

    return max(0.0, min(float(temperature), 1.0))


def get_available_models() -> dict[str, str]:
    """Retourne les modèles actifs possédant un label valide."""
    available_models: dict[str, str] = {}

    for alias, config in MODEL_REGISTRY.items():
        if not isinstance(config, dict):
            raise TypeError(
                f"Configuration invalide pour {alias!r} : "
                f"un dictionnaire était attendu, "
                f"reçu {type(config).__name__}."
            )

        if not config.get("enabled", False):
            continue

        label = config.get("label")

        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Le modèle {alias!r} est actif, "
                "mais il ne possède pas de champ 'label' valide. "
                f"Configuration reçue : {config!r}"
            )

        available_models[alias] = label.strip()

    return available_models


def get_model_options() -> dict[str, str]:
    """Retourne les modèles actifs sous une forme adaptée à l'interface."""
    return {
        alias: f"{config['mode']} - {config['label']}"
        for alias, config in MODEL_REGISTRY.items()
        if config.get("enabled", False)
    }


def get_default_model_alias() -> str:
    """Retourne l'alias du modèle conversationnel utilisé par défaut."""
    get_model_config(DEFAULT_MODEL_ALIAS)
    return DEFAULT_MODEL_ALIAS


def get_model_capabilities(model_alias: str) -> dict[str, Any]:
    """Retourne les capacités déclarées pour un modèle."""
    config = get_model_config(model_alias)

    images = bool(config.get("supports_images", False))
    documents = bool(config.get("supports_documents", False))
    tools = bool(config.get("supports_tools", False))

    return {
        "supports_images": images,
        "supports_documents": documents,
        "supports_tools": tools,
        "accepted_image_formats": (
            set(BEDROCK_IMAGE_FORMATS)
            if images
            else set()
        ),
        "accepted_document_formats": (
            set(BEDROCK_DOCUMENT_FORMATS)
            if documents
            else set()
        ),
    }


def classify_attachment(
    filename: str,
    capabilities: dict[str, Any],
) -> str | None:
    """Classe une pièce jointe selon les capacités du modèle."""
    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension in capabilities.get("accepted_image_formats", set()):
        return "image"

    if extension in capabilities.get("accepted_document_formats", set()):
        return "document"

    return None


MODEL_TIERS: dict[str, dict[str, Any]] = {
    "nano": {
        "label": "Nano (tri / routage)",
        "model_alias": "nova-2-lite",
        "description": (
            "Routage, classification, extraction structurée "
            "et tâches simples à faible ambiguïté."
        ),
        "relative_cost": 1,
        "max_parallel": 4,
    },
    "fast": {
        "label": "Rapide (exécution)",
        "model_alias": "claude-haiku-4.5",
        "description": (
            "Agents spécialisés, transformations courantes, "
            "rédaction courte et résumés."
        ),
        "relative_cost": 5,
        "max_parallel": 3,
    },
    "smart": {
        "label": "Sérieux (code / analyse)",
        "model_alias": "claude-sonnet-5.5",
        "description": (
            "Code, analyse complexe, statistiques, vision "
            "et raisonnement structuré."
        ),
        "relative_cost": 10,
        "max_parallel": 2,
    },
    "expert": {
        "label": "Expert (chef / arbitrage)",
        "model_alias": "claude-opus-5.5",
        "description": (
            "Direction du système, raisonnement de haut niveau, "
            "critique et arbitrage final."
        ),
        "relative_cost": 20,
        "max_parallel": 1,
    },
}


DEFAULT_WORKER_TIER = "fast"
CHIEF_TIER = "expert"


def get_tier(tier_name: str) -> dict[str, Any]:
    """Retourne une copie de la configuration d'un tier."""
    if tier_name not in MODEL_TIERS:
        allowed_tiers = ", ".join(sorted(MODEL_TIERS))
        raise ValueError(
            f"Tier inconnu : {tier_name}. "
            f"Tiers autorisés : {allowed_tiers}"
        )

    return deepcopy(MODEL_TIERS[tier_name])


def resolve_tier_to_model_alias(tier_name: str) -> str:
    """Résout un tier vers un alias de modèle actif."""
    alias = str(get_tier(tier_name)["model_alias"])
    get_model_config(alias)
    return alias


def get_tier_options() -> dict[str, str]:
    """Retourne les tiers sous une forme adaptée à l'interface."""
    return {
        name: config["label"]
        for name, config in MODEL_TIERS.items()
    }


PROMPTING_PROFILE_BY_ALIAS: dict[str, str] = {
    "nova-2-lite": "light",
    "claude-haiku-4.5": "light",
    "claude-sonnet-5.5": "frontier",
    "claude-opus-5.5": "frontier",
    "pixtral-large": "balanced",
}


DEFAULT_PROMPTING_PROFILE = "balanced"

# Haiku 4.5 reste utilisé pour les opérations de reformulation simples.
# Sonnet 5.5 peut être choisi explicitement si la génération de prompt
# nécessite une analyse plus complexe.
DEFAULT_PROMPTER_MODEL_ALIAS = "claude-haiku-4.5"


def get_prompting_profile(model_alias: str) -> str:
    """Retourne le profil de prompting associé au modèle."""
    return PROMPTING_PROFILE_BY_ALIAS.get(
        model_alias,
        DEFAULT_PROMPTING_PROFILE,
    )