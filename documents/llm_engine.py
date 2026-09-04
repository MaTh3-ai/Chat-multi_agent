"""
Création des modèles et agents Amazon Bedrock utilisés
par l'application.

L'authentification est fournie automatiquement par le rôle IAM
de l'environnement SageMaker AI Studio.

L'agent dispose :
- d'outils de consultation du workspace ;
- d'un outil permettant de proposer une création ou une modification ;
- d'aucun outil permettant d'appliquer directement une modification.

Les modifications doivent être approuvées dans l'interface Streamlit.
"""

from typing import Any

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langgraph.prebuilt import create_react_agent

from core.models import (
    DEFAULT_REGION,
    get_default_model_alias,
    get_model_config,
)

from core.memory_tools import MEMORY_TOOLS
from tools.custom_tools import READ_ONLY_TOOLS
from tools.write_tools import WRITE_PROPOSAL_TOOLS


# -------------------------------------------------------------------
# Configuration des outils
# -------------------------------------------------------------------

READ_TOOLS = list(READ_ONLY_TOOLS)

PROPOSAL_TOOLS = list(WRITE_PROPOSAL_TOOLS)

MEMORY_TOOL_LIST = list(MEMORY_TOOLS)

TOOLS = READ_TOOLS + PROPOSAL_TOOLS + MEMORY_TOOL_LIST


# -------------------------------------------------------------------
# Client Amazon Bedrock Runtime
# -------------------------------------------------------------------

def create_bedrock_runtime_client(
    region_name=DEFAULT_REGION,
):
    """
    Crée un client Amazon Bedrock Runtime.

    Boto3 utilise automatiquement les identifiants temporaires
    du rôle IAM associé à SageMaker AI Studio.
    """

    boto_config = Config(
        connect_timeout=15,
        read_timeout=3600,
        retries={
            "max_attempts": 4,
            "mode": "standard",
        },
        tcp_keepalive=True,
    )

    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region_name,
        config=boto_config,
    )


# -------------------------------------------------------------------
# Création d'un modèle de conversation
# -------------------------------------------------------------------

def create_chat_model(
    model_alias=None,
    temperature=None,
    max_tokens=None,
):
    """
    Crée un modèle LangChain utilisant l'API Bedrock Converse.

    Les paramètres transmis à la fonction remplacent les valeurs
    par défaut définies dans le catalogue des modèles.
    """

    selected_alias = (
        model_alias
        if model_alias is not None
        else get_default_model_alias()
    )

    model_config = get_model_config(
        selected_alias
    )

    selected_max_tokens = (
        max_tokens
        if max_tokens is not None
        else model_config["max_tokens"]
    )

    selected_temperature = (
        temperature
        if temperature is not None
        else model_config.get("temperature")
    )

    if not isinstance(selected_max_tokens, int):
        raise ValueError(
            "Le nombre maximal de tokens doit être un entier."
        )

    if selected_max_tokens < 1:
        raise ValueError(
            "Le nombre maximal de tokens doit être positif."
        )

    if (
        selected_temperature is not None
        and not isinstance(
            selected_temperature,
            (int, float),
        )
    ):
        raise ValueError(
            "La température doit être un nombre."
        )

    if (
        selected_temperature is not None
        and not 0.0
        <= selected_temperature
        <= 1.0
    ):
        raise ValueError(
            "La température doit être comprise entre 0 et 1."
        )

    client = create_bedrock_runtime_client(
        region_name=model_config["region"],
    )

    model_arguments = {
        "client": client,
        "model_id": model_config["model_id"],
        "max_tokens": selected_max_tokens,
    }

    if selected_temperature is not None:
        model_arguments["temperature"] = (
            selected_temperature
        )

    return ChatBedrockConverse(
        **model_arguments
    )


# -------------------------------------------------------------------
# Création de l'agent
# -------------------------------------------------------------------

def get_bedrock_agent(
    model_alias=None,
    temperature=None,
    max_tokens=None,
    additional_tools=None,
):
    """
    Crée un agent LangGraph utilisant un modèle Amazon Bedrock.

    L'agent reçoit :
    - les outils de lecture du workspace ;
    - l'outil de proposition de modification ;
    - les éventuels outils supplémentaires fournis explicitement.

    Les outils d'approbation ou de refus d'une modification
    ne sont jamais transmis au modèle.
    """

    model = create_chat_model(
        model_alias=model_alias,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    agent_tools = list(TOOLS)

    if additional_tools:
        for additional_tool in additional_tools:
            if additional_tool not in agent_tools:
                agent_tools.append(
                    additional_tool
                )

    return create_react_agent(
        model=model,
        tools=agent_tools,
    )


# -------------------------------------------------------------------
# Informations sur les outils
# -------------------------------------------------------------------

def get_tool_names():
    """
    Retourne les noms de tous les outils accessibles à l'agent.
    """

    return [
        current_tool.name
        for current_tool in TOOLS
    ]


def get_read_tool_names():
    """
    Retourne les noms des outils de consultation du workspace.
    """

    return [
        current_tool.name
        for current_tool in READ_TOOLS
    ]


def get_proposal_tool_names():
    """
    Retourne les noms des outils permettant de préparer
    une proposition de modification.
    """

    return [
        current_tool.name
        for current_tool in PROPOSAL_TOOLS
    ]


def get_memory_tool_names():
    """
    Retourne les noms des outils de gestion de la mémoire
    durable de la conversation.
    """

    return [
        current_tool.name
        for current_tool in MEMORY_TOOL_LIST
    ]


def get_tool_descriptions():
    """
    Retourne les descriptions de tous les outils.

    Le dictionnaire utilise le nom technique de l'outil comme clé
    et sa description LangChain comme valeur.
    """

    descriptions = {}

    for current_tool in TOOLS:
        descriptions[current_tool.name] = (
            current_tool.description
            or "Aucune description disponible."
        )

    return descriptions


def get_tool_summary():
    """
    Retourne un résumé structuré des outils du moteur.
    """

    return {
        "read_tools": get_read_tool_names(),
        "proposal_tools": get_proposal_tool_names(),
        "all_tools": get_tool_names(),
        "descriptions": get_tool_descriptions(),
    }


# -------------------------------------------------------------------
# Diagnostic du moteur
# -------------------------------------------------------------------

def print_engine_status():
    """
    Affiche les informations principales du moteur.

    Cette fonction ne réalise aucun appel au modèle.
    """

    default_alias = get_default_model_alias()
    default_config = get_model_config(
        default_alias
    )

    print("=" * 70)
    print("Moteur Amazon Bedrock")
    print("=" * 70)

    print(
        "Région par défaut :",
        DEFAULT_REGION,
    )

    print(
        "Alias par défaut :",
        default_alias,
    )

    print(
        "Modèle par défaut :",
        default_config["label"],
    )

    print(
        "Inference Profile ID :",
        default_config["model_id"],
    )

    print()
    print("Outils de lecture :")

    for tool_name in get_read_tool_names():
        print("-", tool_name)

    print()
    print("Outils de proposition :")

    for tool_name in get_proposal_tool_names():
        print("-", tool_name)

    print()
    print(
        "Nombre total d'outils :",
        len(TOOLS),
    )


if __name__ == "__main__":
    print_engine_status()