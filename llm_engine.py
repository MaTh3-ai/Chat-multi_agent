"""Creation des modeles et agents Amazon Bedrock de l'application."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent

from core.models import (
    DEFAULT_REGION,
    get_default_model_alias,
    get_model_config,
    sanitize_temperature,
)
from core.memory_tools import MEMORY_TOOLS
from tools.custom_tools import READ_ONLY_TOOLS
from tools.write_tools import DIRECT_WRITE_TOOLS, WRITE_PROPOSAL_TOOLS

READ_TOOLS = list(READ_ONLY_TOOLS)
PROPOSAL_TOOLS = list(WRITE_PROPOSAL_TOOLS)
MEMORY_TOOL_LIST = list(MEMORY_TOOLS)
TOOLS = READ_TOOLS + PROPOSAL_TOOLS + MEMORY_TOOL_LIST

# Lecteur de workspace dedie au mode multi-agent : uniquement les outils de
# consultation. Aucun outil de proposition, d'approbation, d'ecriture directe
# ou de memoire n'est expose ici. C'est ce jeu qui est confie au Chef pendant
# la phase DISCOVERY.
WORKSPACE_READER_TOOLS = list(READ_ONLY_TOOLS)

# Editeur multi-agent : lecture, modifications directes sauvegardees et
# controles. Le chatbot classique conserve son circuit de propositions.
WORKSPACE_EDITOR_TOOLS = list(READ_ONLY_TOOLS) + list(DIRECT_WRITE_TOOLS)

CONNECT_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "15"))
READ_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_READ_TIMEOUT", "300"))
# Les workers gerent leur propre retry selectif. Le client ne fait qu'un
# retry SDK court afin d'eviter la multiplication silencieuse des tentatives.
SDK_TOTAL_MAX_ATTEMPTS = int(os.getenv("BEDROCK_SDK_TOTAL_MAX_ATTEMPTS", "2"))


def _validate_positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} doit etre un entier positif.")
    return value


@lru_cache(maxsize=8)
def create_bedrock_runtime_client(region_name: str = DEFAULT_REGION):
    """Cree et met en cache un client Bedrock Runtime par region."""
    _validate_positive("BEDROCK_CONNECT_TIMEOUT", CONNECT_TIMEOUT_SECONDS)
    _validate_positive("BEDROCK_READ_TIMEOUT", READ_TIMEOUT_SECONDS)
    _validate_positive("BEDROCK_SDK_TOTAL_MAX_ATTEMPTS", SDK_TOTAL_MAX_ATTEMPTS)
    config = Config(
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        read_timeout=READ_TIMEOUT_SECONDS,
        retries={"total_max_attempts": SDK_TOTAL_MAX_ATTEMPTS, "mode": "standard"},
        tcp_keepalive=True,
    )
    return boto3.client("bedrock-runtime", region_name=region_name, config=config)


def create_chat_model(
    model_alias: str | None = None,
    temperature: float | int | None = None,
    max_tokens: int | None = None,
):
    """Construit ChatBedrockConverse a partir du registre central."""
    alias = model_alias or get_default_model_alias()
    model_config = get_model_config(alias)
    selected_max_tokens = max_tokens if max_tokens is not None else model_config["max_tokens"]
    _validate_positive("max_tokens", selected_max_tokens)
    selected_temperature = sanitize_temperature(alias, temperature)

    arguments: dict[str, Any] = {
        "client": create_bedrock_runtime_client(model_config["region"]),
        "model_id": model_config["model_id"],
        "max_tokens": selected_max_tokens,
    }
    if selected_temperature is not None:
        arguments["temperature"] = selected_temperature
    return ChatBedrockConverse(**arguments)


def _deduplicate_tools(tools: Iterable[Any]) -> list[Any]:
    """Dedoublonne les outils par nom technique, avec echec explicite."""
    unique: dict[str, Any] = {}
    for current_tool in tools:
        name = getattr(current_tool, "name", None)
        if not name:
            raise ValueError("Un outil sans nom technique a ete fourni.")
        if name in unique and unique[name] is not current_tool:
            raise ValueError(f"Deux outils differents portent le nom {name!r}.")
        unique[name] = current_tool
    return list(unique.values())


def get_bedrock_agent(
    model_alias: str | None = None,
    temperature: float | int | None = None,
    max_tokens: int | None = None,
    additional_tools: Iterable[Any] | None = None,
):
    """Cree l'agent de chat. Aucun outil d'approbation ne lui est expose."""
    model = create_chat_model(model_alias, temperature, max_tokens)
    tools = _deduplicate_tools([*TOOLS, *(additional_tools or [])])
    return create_react_agent(model=model, tools=tools)


def get_workspace_reader_agent(
    model_alias: str | None = None,
    temperature: float | int | None = None,
    max_tokens: int | None = None,
):
    """Cree un agent LangGraph de LECTURE SEULE du workspace.

    Cet agent n'expose QUE les outils de consultation (WORKSPACE_READER_TOOLS) :
    informations_workspace, lister_fichiers, informations_fichier,
    explorer_workspace_cible, lire_fichier, rechercher_dans_fichiers.

    Il ne recoit volontairement aucun outil de proposition, d'approbation,
    d'ecriture directe ni de memoire : il ne peut qu'observer le workspace.

    Note d'architecture : la phase DISCOVERY du mode multi-agent est desormais
    DETERMINISTE (l'orchestrateur appelle directement les outils Python de
    consultation, voir core.multi_agent.orchestrator.ChefDeProjet.discover).
    Le Chef n'utilise donc pas cet agent LLM. Cette fabrique est conservee pour
    les usages hors mode multi-agent et pour les tests.
    """
    model = create_chat_model(model_alias, temperature, max_tokens)
    tools = _deduplicate_tools(WORKSPACE_READER_TOOLS)
    return create_react_agent(model=model, tools=tools)


def get_workspace_editor_agent(
    model_alias: str | None = None,
    temperature: float | int | None = None,
    max_tokens: int | None = None,
):
    """Cree un agent multi-agent de lecture et d'ecriture directe.

    Les écritures restent confinées au workspace, sont validées, sauvegardées
    et appliquées immédiatement. Le chatbot classique continue d'utiliser ses
    propres outils de proposition soumis à approbation.
    """
    model = create_chat_model(model_alias, temperature, max_tokens)
    tools = _deduplicate_tools(WORKSPACE_EDITOR_TOOLS)
    return create_react_agent(model=model, tools=tools)


def get_read_only_tools() -> list[Any]:
    """Retourne la liste des outils de consultation, pour usage direct/tests."""
    return list(WORKSPACE_READER_TOOLS)


def get_workspace_reader_tool_names() -> list[str]:
    return [tool.name for tool in WORKSPACE_READER_TOOLS]


def get_tool_names() -> list[str]:
    return [tool.name for tool in TOOLS]


def get_read_tool_names() -> list[str]:
    return [tool.name for tool in READ_TOOLS]


def get_proposal_tool_names() -> list[str]:
    return [tool.name for tool in PROPOSAL_TOOLS]


def get_memory_tool_names() -> list[str]:
    return [tool.name for tool in MEMORY_TOOL_LIST]


def get_tool_descriptions() -> dict[str, str]:
    return {tool.name: tool.description or "Aucune description disponible." for tool in TOOLS}


def get_tool_summary() -> dict[str, Any]:
    return {
        "read_tools": get_read_tool_names(),
        "proposal_tools": get_proposal_tool_names(),
        "memory_tools": get_memory_tool_names(),
        "all_tools": get_tool_names(),
        "descriptions": get_tool_descriptions(),
    }


def print_engine_status() -> None:
    alias = get_default_model_alias()
    config = get_model_config(alias)
    print("=" * 70)
    print("Moteur Amazon Bedrock")
    print("Region par defaut :", DEFAULT_REGION)
    print("Alias par defaut :", alias)
    print("Modele par defaut :", config["label"])
    print("Inference Profile ID :", config["model_id"])
    print("Outils :", ", ".join(get_tool_names()))


if __name__ == "__main__":
    print_engine_status()
