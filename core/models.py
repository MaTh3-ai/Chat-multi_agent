"""Catalogue central des modeles Amazon Bedrock autorises."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_MODEL_ALIAS = "claude-sonnet-4.5"
DEFAULT_REGION = "eu-west-3"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "claude-sonnet-4.5": {"label": "Claude Sonnet 4.5", "mode": "Equilibre", "description": "Modele polyvalent pour conversation, code et analyse.", "model_id": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "region": DEFAULT_REGION, "provider": "anthropic", "max_tokens": 16384, "temperature": 0.2, "supports_images": True, "supports_documents": True, "supports_tools": True, "enabled": True},
    "claude-opus-4.7": {"label": "Claude Opus 4.7", "mode": "Approfondi", "description": "Modele avance pour les raisonnements complexes.", "model_id": "eu.anthropic.claude-opus-4-7", "region": DEFAULT_REGION, "provider": "anthropic", "max_tokens": 16384, "temperature": None, "supports_images": True, "supports_documents": True, "supports_tools": True, "enabled": True},
    "claude-opus-4.8": {"label": "Claude Opus 4.8", "mode": "Expert", "description": "Modele expert pour l'arbitrage, la critique et la synthese.", "model_id": "eu.anthropic.claude-opus-4-8", "region": DEFAULT_REGION, "provider": "anthropic", "max_tokens": 16384, "temperature": None, "supports_images": True, "supports_documents": True, "supports_tools": True, "enabled": True},
    "claude-haiku-4.5": {"label": "Claude Haiku 4.5", "mode": "Rapide", "description": "Modele economique pour execution, redaction et resumes.", "model_id": "eu.anthropic.claude-haiku-4-5-20251001-v1:0", "region": DEFAULT_REGION, "provider": "anthropic", "max_tokens": 8192, "temperature": 0.2, "supports_images": True, "supports_documents": True, "supports_tools": True, "enabled": True},
    "claude-3-haiku": {"label": "Claude 3 Haiku", "mode": "Rapide", "description": "Ancien modele economique pour les taches legeres.", "model_id": "eu.anthropic.claude-3-haiku-20240307-v1:0", "region": DEFAULT_REGION, "provider": "anthropic", "max_tokens": 4096, "temperature": 0.2, "supports_images": True, "supports_documents": True, "supports_tools": True, "enabled": True},
    "nova-lite": {"label": "Nova Lite", "mode": "Rapide", "description": "Modele Amazon economique et multimodal.", "model_id": "eu.amazon.nova-lite-v1:0", "region": DEFAULT_REGION, "provider": "amazon", "max_tokens": 4096, "temperature": 0.2, "supports_images": True, "supports_documents": False, "supports_tools": True, "enabled": True},
    "nova-micro": {"label": "Nova Micro", "mode": "Nano", "description": "Modele texte economique pour tri et routing.", "model_id": "eu.amazon.nova-micro-v1:0", "region": DEFAULT_REGION, "provider": "amazon", "max_tokens": 4096, "temperature": 0.2, "supports_images": False, "supports_documents": False, "supports_tools": True, "enabled": True},
}

BEDROCK_IMAGE_FORMATS = frozenset({"png", "jpeg", "jpg", "gif", "webp"})
BEDROCK_DOCUMENT_FORMATS = frozenset({"pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"})


def sanitize_temperature(model_alias: str, temperature: float | int | None) -> float | None:
    """Retourne une temperature valide ou None si le modele n'en accepte pas."""
    config = get_model_config(model_alias)
    if config.get("temperature") is None:
        return None
    if temperature is None:
        temperature = config.get("temperature")
    if temperature is None:
        return None
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("La temperature doit etre un nombre ou None.")
    return float(max(0.0, min(float(temperature), 1.0)))


def get_model_config(model_alias: str) -> dict[str, Any]:
    if model_alias not in MODEL_REGISTRY:
        raise ValueError(f"Modele inconnu : {model_alias}. Modeles declares : {', '.join(sorted(MODEL_REGISTRY))}")
    config = deepcopy(MODEL_REGISTRY[model_alias])
    if not config.get("enabled", False):
        raise ValueError(f"Le modele {config['label']} est desactive.")
    model_id = str(config.get("model_id", ""))
    if not model_id:
        raise ValueError(f"Le modele {config['label']} ne possede pas d'ID.")
    if not model_id.startswith("eu."):
        raise ValueError("Seuls les profils europeens commencant par 'eu.' sont autorises.")
    return config


def get_available_models() -> dict[str, str]:
    return {alias: cfg["label"] for alias, cfg in MODEL_REGISTRY.items() if cfg.get("enabled", False)}


def get_model_options() -> dict[str, str]:
    return {alias: f"{cfg['mode']} - {cfg['label']}" for alias, cfg in MODEL_REGISTRY.items() if cfg.get("enabled", False)}


def get_default_model_alias() -> str:
    get_model_config(DEFAULT_MODEL_ALIAS)
    return DEFAULT_MODEL_ALIAS


def get_model_capabilities(model_alias: str) -> dict[str, Any]:
    config = get_model_config(model_alias)
    images = bool(config.get("supports_images", False))
    documents = bool(config.get("supports_documents", False))
    return {"supports_images": images, "supports_documents": documents, "accepted_image_formats": set(BEDROCK_IMAGE_FORMATS) if images else set(), "accepted_document_formats": set(BEDROCK_DOCUMENT_FORMATS) if documents else set()}


def classify_attachment(filename: str, capabilities: dict[str, Any]) -> str | None:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension in capabilities.get("accepted_image_formats", set()):
        return "image"
    if extension in capabilities.get("accepted_document_formats", set()):
        return "document"
    return None


MODEL_TIERS: dict[str, dict[str, Any]] = {
    "nano": {"label": "Nano (tri / routing)", "model_alias": "nova-micro", "description": "Classification, routing et taches triviales.", "relative_cost": 1, "max_parallel": 2},
    "fast": {"label": "Rapide (execution)", "model_alias": "claude-haiku-4.5", "description": "Execution, redaction et resumes.", "relative_cost": 4, "max_parallel": 2},
    "smart": {"label": "Serieux (code / analyse)", "model_alias": "claude-sonnet-4.5", "description": "Code, analyse et raisonnement structure.", "relative_cost": 15, "max_parallel": 2},
    "expert": {"label": "Expert (chef / arbitrage)", "model_alias": "claude-opus-4.8", "description": "Chef de projet, critique et arbitrage.", "relative_cost": 60, "max_parallel": 1},
}
DEFAULT_WORKER_TIER = "fast"
CHIEF_TIER = "expert"


def get_tier(tier_name: str) -> dict[str, Any]:
    if tier_name not in MODEL_TIERS:
        raise ValueError(f"Tier inconnu : {tier_name}. Tiers autorises : {', '.join(sorted(MODEL_TIERS))}")
    return deepcopy(MODEL_TIERS[tier_name])


def resolve_tier_to_model_alias(tier_name: str) -> str:
    alias = str(get_tier(tier_name)["model_alias"])
    get_model_config(alias)
    return alias


def get_tier_options() -> dict[str, str]:
    return {name: cfg["label"] for name, cfg in MODEL_TIERS.items()}


PROMPTING_PROFILE_BY_ALIAS = {"nova-micro": "light", "nova-lite": "light", "claude-3-haiku": "light", "claude-haiku-4.5": "light", "claude-sonnet-4.5": "balanced", "claude-opus-4.7": "frontier", "claude-opus-4.8": "frontier"}
DEFAULT_PROMPTING_PROFILE = "balanced"
DEFAULT_PROMPTER_MODEL_ALIAS = "claude-haiku-4.5"


def get_prompting_profile(model_alias: str) -> str:
    return PROMPTING_PROFILE_BY_ALIAS.get(model_alias, DEFAULT_PROMPTING_PROFILE)
