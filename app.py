"""
Interface Streamlit de l'assistant personnel Amazon Bedrock.

Fonctionnalités :
- sélection du modèle ;
- conversation avec historique ;
- instruction système personnalisable ;
- consultation sécurisée du workspace ;
- propositions de modifications complètes ou ciblées ;
- aperçu des différences ;
- validation syntaxique ;
- approbation humaine avant écriture ;
- sauvegarde automatique ;
- affichage des tokens et appels d'outils.
"""

import json

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

from core.models import (
    get_available_models,
    get_default_model_alias,
    get_model_capabilities,
    classify_attachment,
)

from core import conversation_store
from core.memory_tools import (
    empty_memory,
    format_memory_block,
    get_memory_state,
    set_memory_state,
    apply_sliding_window,
)

from llm_engine import (
    get_bedrock_agent,
    get_memory_tool_names,
    get_proposal_tool_names,
    get_read_tool_names,
    get_tool_names,
)

from tools.write_tools import (
    FileModificationError,
    approve_proposal,
    clear_resolved_proposals,
    get_pending_proposals,
    reject_proposal,
)
from core.prompt_rewriter import rewrite_user_prompt
from core.multi_agent.ui import render_multi_agent


# -------------------------------------------------------------------
# Configuration générale
# -------------------------------------------------------------------

APP_TITLE = "Assistant IA personnel"
# Limite de recursion LangGraph pour la boucle de CHAT (agent principal).
# Conservee a 60 : la baisser augmenterait le risque de GraphRecursionError.
CHAT_RECURSION_LIMIT = 60

DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant IA personnel polyvalent. "
    "Tu aides l'utilisateur à réfléchir, analyser, développer, "
    "critiquer et faire progresser différents projets. "
    "Réponds de manière claire, précise et structurée. "
    "Si une information te manque, indique-le explicitement. "
    "\n\n"
    "ACCÈS RÉEL AU WORKSPACE : Tu disposes RÉELLEMENT d'outils de "
    "consultation et de modification du workspace. Ces outils sont "
    "ACTIFS et FONCTIONNELS. Ne demande JAMAIS à l'utilisateur de "
    "confirmer que les outils sont disponibles, de coller le contenu "
    "de fichiers accessibles, ou de fournir des informations que tu "
    "peux obtenir toi-même. Utilise les outils IMMÉDIATEMENT pour "
    "répondre aux demandes d'audit, d'analyse ou de modification. "
    "\n\n"
    "Le workspace contient des fichiers avec lesquels tu peux "
    "travailler. Il peut notamment contenir une copie bêta de "
    "l'application utilisée pour tester tes propres améliorations. "
    "Les chemins transmis aux outils sont toujours relatifs au "
    "workspace. Ne demande jamais de chemin absolu. "
    "\n\n"
    "CONSULTATION DES FICHIERS : "
    "utilise informations_workspace pour examiner le workspace, "
    "lister_fichiers pour explorer les dossiers, "
    "informations_fichier pour consulter les métadonnées, "
    "rechercher_dans_fichiers pour localiser un symbole ou un passage "
    "et lire_fichier pour lire uniquement les extraits nécessaires. "
    "Ne prétends jamais connaître un fichier sans l'avoir consulté. "
    "\n\n"
    "MODIFICATIONS CIBLEES : "
    "pour corriger une fonction, un import, une configuration ou "
    "quelques lignes, utilise proposer_remplacement_cible. "
    "Le paramètre ancien_texte doit reprendre exactement le passage "
    "existant et nouveau_texte doit contenir son remplacement. "
    "\n\n"
    "Pour ajouter du contenu avant ou après une ancre unique, utilise "
    "proposer_insertion_ciblee. Choisis une ancre précise qui "
    "n'apparaît qu'une fois dans le fichier. "
    "\n\n"
    "Pour retirer un bloc précis sans supprimer le fichier, utilise "
    "proposer_suppression_ciblee. Le texte à retirer doit apparaître "
    "exactement une fois. "
    "\n\n"
    "Utilise proposer_modification_fichier uniquement pour créer un "
    "nouveau fichier, remplacer un petit fichier ou effectuer une "
    "réécriture complète réellement nécessaire. Ne reconstruis pas "
    "un fichier volumineux pour modifier seulement quelques lignes. "
    "\n\n"
    "Avant chaque proposition, recherche puis lis uniquement la zone "
    "pertinente. Evite de lire plusieurs fois le même contenu. "
    "Si plusieurs corrections indépendantes sont nécessaires, prépare "
    "plusieurs propositions ciblées clairement compréhensibles. "
    "\n\n"
    "DISTINCTION IMPORTANTE : les outils de custom_tools sont non "
    "destructifs, mais cela ne signifie pas que le workspace est monté "
    "en lecture seule. Fie-toi au champ 'Mode d'accès réel' de "
    "informations_workspace. Si l'écriture après approbation est possible, "
    "appelle les outils proposer_* lorsqu'une modification est demandée. "
    "Ne confonds pas création d'une proposition et application au fichier. "
    "\n\n"
    "Une proposition n'est jamais appliquée automatiquement. "
    "Après avoir créé une proposition, indique qu'elle attend "
    "l'approbation de l'utilisateur dans l'interface. "
    "Ne prétends jamais qu'un fichier a été modifié tant que "
    "l'utilisateur n'a pas approuvé la proposition. "
    "Tu ne peux pas appliquer directement une proposition, supprimer "
    "un fichier entier ou exécuter une commande système."
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Fonctions associées aux messages
# -------------------------------------------------------------------

def extract_response_text(content):
    """
    Extrait le texte d'une réponse LangChain.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []

        for block in content:
            if isinstance(block, dict):
                # On ne garde que les blocs textuels ;
                # les blocs techniques (tool_use, reasoning...)
                # ne sont pas renvoyés à l'utilisateur.
                block_type = block.get("type")

                if block_type in (None, "text"):
                    text = block.get("text")

                    if text:
                        texts.append(str(text))

            elif isinstance(block, str):
                if block.strip():
                    texts.append(block)

        return "\n".join(texts)

    return str(content)


def build_attachment_blocks(attachments):
    """
    Convertit les pièces jointes du tour courant en blocs
    multimodaux au format Amazon Bedrock Converse.

    Chaque pièce jointe est un dictionnaire :
    - "kind"   : "image" ou "document" ;
    - "name"   : nom de fichier d'origine ;
    - "format" : format normalisé (ex. "jpeg", "png", "pdf") ;
    - "data"   : contenu binaire brut (bytes).

    Retourne une liste de blocs compatibles Converse :
    - image    -> {"image": {"format": ..., "source": {"bytes": ...}}}
    - document -> {"document": {"format": ..., "name": ...,
                                "source": {"bytes": ...}}}

    Remarque : le champ "source.bytes" reçoit les octets bruts.
    langchain_aws (Bedrock Converse) se charge de l'encodage
    base64 au moment de l'appel API. Si une version de la
    bibliothèque exigeait une chaîne base64, il suffirait de
    remplacer "raw" par base64.b64encode(raw).decode("utf-8").
    """

    blocks = []

    for attachment in attachments or []:
        kind = attachment.get("kind")
        raw = attachment.get("data")
        file_format = attachment.get("format")

        if not raw or not file_format:
            continue

        if kind == "image":
            blocks.append(
                {
                    "image": {
                        "format": file_format,
                        "source": {"bytes": raw},
                    }
                }
            )

        elif kind == "document":
            # Bedrock impose un nom de document non vide et
            # limité à certains caractères ; on retombe sur un
            # nom neutre si nécessaire.
            doc_name = attachment.get("name") or "document"

            blocks.append(
                {
                    "document": {
                        "format": file_format,
                        "name": doc_name,
                        "source": {"bytes": raw},
                    }
                }
            )

    return blocks


def build_langchain_messages(messages, system_prompt):
    """
    Convertit l'historique Streamlit en messages LangChain.

    Deux optimisations de tokens sont appliquées :
    - la mémoire compacte de la conversation (résumé, tâches,
      décisions, fichiers) est injectée en tête du message système,
      afin de conserver le contexte durable sans renvoyer tout
      l'historique ;
    - une fenêtre glissante ne conserve que les derniers messages
      bruts renvoyés au modèle.
    """

    memory_block = format_memory_block(
        get_memory_state()
    )

    if memory_block:
        combined_system_prompt = (
            f"{system_prompt}\n\n{memory_block}"
        )

    else:
        combined_system_prompt = system_prompt

    langchain_messages = [
        SystemMessage(content=combined_system_prompt)
    ]

    windowed_messages = apply_sliding_window(messages)

    # Les pièces jointes du tour courant (octets bruts) ne sont
    # présentes qu'en session : elles doivent être injectées sur le
    # DERNIER message utilisateur, sous forme de blocs multimodaux.
    current_attachments = st.session_state.get(
        "current_attachments", []
    )

    last_user_index = None

    for index, message in enumerate(windowed_messages):
        if message.get("role") == "user":
            last_user_index = index

    for index, message in enumerate(windowed_messages):
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            is_last_user = index == last_user_index

            if is_last_user and current_attachments:
                blocks = build_attachment_blocks(
                    current_attachments
                )

                # Le texte reste un bloc à part entière. On combine
                # les blocs multimodaux et le texte dans une liste.
                combined = list(blocks)

                if content:
                    combined.append(
                        {"text": str(content)}
                    )

                langchain_messages.append(
                    HumanMessage(content=combined)
                )

            else:
                langchain_messages.append(
                    HumanMessage(content=content)
                )

        elif role == "assistant":
            langchain_messages.append(
                AIMessage(content=content)
            )

    return langchain_messages


def find_last_ai_message(agent_messages):
    """
    Recherche le dernier message textuel produit par l'agent.
    """

    for message in reversed(agent_messages):
        if not isinstance(message, AIMessage):
            continue

        content = extract_response_text(
            message.content
        )

        if content.strip():
            return message

    return None


def collect_usage_metadata(agent_messages):
    """
    Additionne les tokens consommés par tous les appels au modèle.
    """

    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    for message in agent_messages:
        if not isinstance(message, AIMessage):
            continue

        usage = getattr(
            message,
            "usage_metadata",
            None,
        )

        if not usage:
            continue

        total_usage["input_tokens"] += (
            usage.get("input_tokens", 0) or 0
        )

        total_usage["output_tokens"] += (
            usage.get("output_tokens", 0) or 0
        )

        total_usage["total_tokens"] += (
            usage.get("total_tokens", 0) or 0
        )

    return total_usage


def count_tool_calls(agent_messages):
    """
    Compte tous les appels d'outils demandés par le modèle.
    """

    tool_call_count = 0

    for message in agent_messages:
        if not isinstance(message, AIMessage):
            continue

        tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if tool_calls:
            tool_call_count += len(tool_calls)

    return tool_call_count


def collect_tool_call_names(agent_messages):
    """Retourne les noms des outils réellement demandés par le modèle."""
    names = []
    for message in agent_messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            name = (tool_call.get("name") if isinstance(tool_call, dict)
                    else getattr(tool_call, "name", None))
            if name:
                names.append(str(name))
    return names


def collect_tool_results(agent_messages):
    """Collecte les résultats d'outils pour distinguer succès et refus."""
    results = []
    for message in agent_messages:
        if not isinstance(message, ToolMessage):
            continue
        results.append({
            "name": getattr(message, "name", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "content": extract_response_text(message.content).strip(),
        })
    return results



# -------------------------------------------------------------------
# Fonctions d'affichage
# -------------------------------------------------------------------

def display_usage_information(
    usage,
    tool_call_count,
):
    """
    Affiche les informations techniques d'une réponse.
    """

    with st.expander(
        "Informations techniques",
        expanded=False,
    ):
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Tokens d'entrée",
            usage.get("input_tokens", 0),
        )

        col2.metric(
            "Tokens de sortie",
            usage.get("output_tokens", 0),
        )

        col3.metric(
            "Total",
            usage.get("total_tokens", 0),
        )

        col4.metric(
            "Appels d'outils",
            tool_call_count,
        )


def add_system_notification(content):
    """
    Ajoute une notification à l'historique.
    """

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "model_label": "Système de fichiers",
            "usage": {},
            "tool_call_count": 0,
        }
    )


def get_operation_label(operation):
    """
    Retourne le nom lisible d'un type de proposition.
    """

    labels = {
        "creation_complete": "Création complète",
        "remplacement_complet": "Remplacement complet",
        "remplacement_cible": "Remplacement ciblé",
        "insertion_avant": "Insertion avant une ancre",
        "insertion_apres": "Insertion après une ancre",
        "suppression_ciblee": "Suppression ciblée",
    }

    return labels.get(
        operation,
        operation.replace(
            "_",
            " ",
        ).capitalize(),
    )


def get_code_language(relative_path):
    """
    Détermine le langage utilisé pour l'aperçu Streamlit.
    """

    if "." not in relative_path:
        return "text"

    extension = relative_path.rsplit(
        ".",
        maxsplit=1,
    )[-1].lower()

    language_mapping = {
        "py": "python",
        "md": "markdown",
        "json": "json",
        "jsonl": "json",
        "toml": "toml",
        "yaml": "yaml",
        "yml": "yaml",
        "html": "html",
        "css": "css",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "sql": "sql",
        "sh": "bash",
        "ps1": "powershell",
        "csv": "csv",
        "txt": "text",
    }

    return language_mapping.get(
        extension,
        "text",
    )


def display_syntax_validation(proposal):
    """
    Affiche le résultat de la validation syntaxique.
    """

    validation = proposal.get(
        "syntax_validation",
        {},
    )

    if not validation:
        return True

    checked = validation.get(
        "checked",
        False,
    )

    valid = validation.get(
        "valid",
        True,
    )

    message = validation.get(
        "message",
        "Aucune information disponible.",
    )

    if not checked:
        st.info(message)
        return True

    if valid:
        st.success(
            f"Validation syntaxique réussie : {message}"
        )
        return True

    st.error(
        f"Validation syntaxique échouée : {message}"
    )

    return False


def display_patch_metadata(proposal):
    """
    Affiche les informations techniques du patch ciblé.
    """

    metadata = proposal.get(
        "metadata",
        {},
    )

    if not metadata:
        return

    patch_type = metadata.get(
        "patch_type"
    )

    if patch_type:
        st.caption(
            f"Type technique du patch : {patch_type}"
        )

    position = metadata.get(
        "position"
    )

    if position:
        st.caption(
            f"Position de l'insertion : {position}"
        )


def display_pending_proposals():
    """
    Affiche les propositions en attente avec approbation ou refus.
    """

    pending_proposals = (
        get_pending_proposals()
    )

    if not pending_proposals:
        return

    st.subheader(
        "📝 Modifications en attente d'approbation"
    )

    st.warning(
        "Aucun fichier n'est modifié tant que tu n'as pas "
        "cliqué sur Approuver."
    )

    for proposal in pending_proposals:
        proposal_id = proposal[
            "proposal_id"
        ]

        relative_path = proposal[
            "path"
        ]

        operation = proposal[
            "operation"
        ]

        operation_label = get_operation_label(
            operation
        )

        justification = proposal.get(
            "justification",
            "Aucune justification fournie.",
        )

        diff_preview = proposal.get(
            "diff",
            "Aucun aperçu disponible.",
        )

        old_content = proposal.get(
            "old_content",
            "",
        )

        new_content = proposal.get(
            "new_content",
            "",
        )

        title = (
            f"{operation_label} : {relative_path}"
        )

        with st.expander(
            title,
            expanded=True,
        ):
            st.markdown(
                f"**Fichier :** `{relative_path}`"
            )

            st.markdown(
                f"**Opération :** {operation_label}"
            )

            st.markdown(
                f"**Justification :** {justification}"
            )

            st.caption(
                f"Identifiant : {proposal_id}"
            )

            syntax_valid = (
                display_syntax_validation(
                    proposal
                )
            )

            display_patch_metadata(
                proposal
            )

            approve_column, reject_column = (
                st.columns(2)
            )

            with approve_column:
                approve_clicked = st.button(
                    "✅ Approuver",
                    key=f"approve_{proposal_id}",
                    type="primary",
                    use_container_width=True,
                    disabled=not syntax_valid,
                )

            with reject_column:
                reject_clicked = st.button(
                    "❌ Refuser",
                    key=f"reject_{proposal_id}",
                    use_container_width=True,
                )

            if approve_clicked:
                try:
                    result = approve_proposal(
                        proposal_id
                    )

                    backup_path = result.get(
                        "backup_path"
                    )

                    notification = (
                        f"✅ La proposition pour "
                        f"`{relative_path}` a été approuvée "
                        "et appliquée."
                    )

                    if backup_path:
                        notification += (
                            "\n\nSauvegarde créée : "
                            f"`{backup_path}`"
                        )

                    add_system_notification(
                        notification
                    )

                    clear_resolved_proposals()

                    st.rerun()

                except FileModificationError as error:
                    st.error(
                        "Impossible d'appliquer la "
                        f"modification : {error}"
                    )

                except Exception as error:
                    st.exception(error)

            if reject_clicked:
                try:
                    reject_proposal(
                        proposal_id
                    )

                    add_system_notification(
                        f"❌ La proposition pour "
                        f"`{relative_path}` a été refusée. "
                        "Aucun fichier n'a été modifié."
                    )

                    clear_resolved_proposals()

                    st.rerun()

                except FileModificationError as error:
                    st.error(
                        "Impossible de refuser la "
                        f"proposition : {error}"
                    )

                except Exception as error:
                    st.exception(error)

    st.divider()


# -------------------------------------------------------------------
# Chargement et cache de l'agent
# -------------------------------------------------------------------

@st.cache_resource
def load_agent(model_alias):
    """
    Charge un agent pour un modèle donné.
    
    Cet agent dispose RÉELLEMENT des outils de consultation et de proposition.
    Le prompt système DEFAULT_SYSTEM_PROMPT lui indique explicitement de les
    utiliser activement pour répondre aux demandes de l'utilisateur.
    """

    return get_bedrock_agent(
        model_alias=model_alias
    )


def reset_conversation():
    """
    Réinitialise l'historique de la conversation.

    Les propositions en attente et les fichiers sont conservés.
    """

    st.session_state.messages = []
    st.session_state.memory = empty_memory()
    st.session_state.conversation_id = (
        conversation_store.new_conversation_id()
    )


def persist_current_conversation():
    """
    Sauvegarde la conversation courante sur le disque.

    Ne fait rien si aucun message utile n'existe encore.
    """

    messages = st.session_state.get("messages", [])

    if not messages:
        return

    conversation_id = st.session_state.get("conversation_id")

    if not conversation_id:
        conversation_id = conversation_store.new_conversation_id()
        st.session_state.conversation_id = conversation_id

    try:
        conversation_store.save_conversation(
            conversation_id=conversation_id,
            messages=messages,
            memory=st.session_state.get("memory"),
        )
    except conversation_store.ConversationStoreError:
        # La persistance ne doit jamais interrompre l'application.
        pass


# -------------------------------------------------------------------
# Initialisation de la session
# -------------------------------------------------------------------

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        DEFAULT_SYSTEM_PROMPT
    )

# Restauration automatique de la dernière conversation au démarrage.
# Objectif du stockage : survivre à un redémarrage de l'application.
# Sans ce bloc, la session repart toujours vide et la mémoire
# persistée n'est jamais réinjectée tant que l'utilisateur ne
# reclique pas manuellement sur une conversation.
if "conversation_id" not in st.session_state:
    _latest_summaries = conversation_store.list_conversations()
    _restored = None

    if _latest_summaries:
        _restored = conversation_store.load_conversation(
            _latest_summaries[0]["id"]
        )

    if _restored is not None:
        st.session_state.conversation_id = _restored["id"]
        st.session_state.messages = _restored.get("messages", [])
        st.session_state.memory = _restored.get(
            "memory",
            empty_memory(),
        )
    else:
        st.session_state.conversation_id = (
            conversation_store.new_conversation_id()
        )
        st.session_state.messages = []
        st.session_state.memory = empty_memory()

# Filets de sécurité si la restauration n'a pas renseigné ces clés.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = empty_memory()

# -------------------------------------------------------------------
# États de requête API pour éviter les interruptions lors des reruns
# -------------------------------------------------------------------

# Initialisation des états de requête API si absents
if "api_request_status" not in st.session_state:
    # Valeurs possibles : "idle", "pending", "success", "error"
    st.session_state.api_request_status = "idle"

if "api_request_error" not in st.session_state:
    st.session_state.api_request_error = None

if "api_request_model" not in st.session_state:
    # Modèle utilisé pour la requête en cours (pour éviter les changements)
    st.session_state.api_request_model = None

if "api_request_operation" not in st.session_state:
    # Type d'opération en cours : "chat", "rewrite", None
    st.session_state.api_request_operation = None


# -------------------------------------------------------------------
# Fonctions de gestion d'état API
# -------------------------------------------------------------------

def begin_api_request(operation: str, model: str) -> None:
    """Démarre une requête API et verrouille l'interface."""
    st.session_state.api_request_status = "pending"
    st.session_state.api_request_operation = operation
    st.session_state.api_request_model = model
    st.session_state.api_request_error = None


def complete_api_request() -> None:
    """Termine une requête API avec succès."""
    st.session_state.api_request_status = "success"
    st.session_state.api_request_operation = None
    st.session_state.api_request_model = None
    st.session_state.api_request_error = None


def fail_api_request(error: str) -> None:
    """Termine une requête API avec une erreur."""
    st.session_state.api_request_status = "error"
    st.session_state.api_request_error = error
    st.session_state.api_request_operation = None
    st.session_state.api_request_model = None


def reset_api_request() -> None:
    """Réinitialise l'état de la requête API."""
    st.session_state.api_request_status = "idle"
    st.session_state.api_request_operation = None
    st.session_state.api_request_model = None
    st.session_state.api_request_error = None


def is_api_request_pending() -> bool:
    """Vérifie si une requête API est en cours."""
    return st.session_state.api_request_status == "pending"



# -------------------------------------------------------------------
# Chargement des modèles
# -------------------------------------------------------------------

available_models = get_available_models()

if not available_models:
    st.error(
        "Aucun modèle Bedrock n'est disponible."
    )

    st.stop()

model_aliases = list(
    available_models.keys()
)

default_model_alias = (
    get_default_model_alias()
)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = (
        default_model_alias
    )

if (
    st.session_state.selected_model
    not in model_aliases
):
    st.session_state.selected_model = (
        default_model_alias
    )


# -------------------------------------------------------------------
# Barre latérale
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    # Verrouillage global de la barre latérale pendant un appel modèle :
    # changer de mode ou de modèle en cours d'appel provoquait un rerun
    # Streamlit qui interrompait le traitement en cours.
    is_locked = is_api_request_pending()

    # Sélecteur de mode : Chat classique ou onglet Multi-agent.
    # La valeur est lue plus bas (st.session_state.app_mode) pour
    # aiguiller vers render_multi_agent().
    app_mode = st.radio(
        "Mode",
        options=["💬 Chat", "🧑‍🤝‍🧑 Multi-agent"],
        index=(
            1
            if st.session_state.get("app_mode")
            == "🧑‍🤝‍🧑 Multi-agent"
            else 0
        ),
        horizontal=True,
        disabled=is_locked,
        help="Désactivé pendant une requête API" if is_locked else None,
    )

    st.session_state.app_mode = app_mode

    st.divider()

    selected_model = st.selectbox(
        "Modèle",
        options=model_aliases,
        index=model_aliases.index(
            st.session_state.selected_model
        ),
        format_func=lambda alias: (
            available_models[alias]
        ),
        disabled=is_locked,
        help="Désactivé pendant une requête API" if is_locked else None,
    )

    st.session_state.selected_model = (
        selected_model
    )

    st.caption(
        f"Modèle actif : "
        f"**{available_models[selected_model]}**"
    )
    
    # Affichage de l'état de la requête API
    if is_locked:
        operation = st.session_state.api_request_operation or "opération"
        model = st.session_state.api_request_model or "modèle"
        st.info(f"🔄 {operation.capitalize()} en cours avec {model}...")
    elif st.session_state.api_request_status == "error":
        st.error(f"❌ Erreur : {st.session_state.api_request_error}")
        if st.button("Réinitialiser", key="reset_api_error"):
            reset_api_request()
            st.rerun()
    if app_mode == "🧑‍🤝‍🧑 Multi-agent":
        st.caption(
            "Le mode multi-agent choisit automatiquement les modèles selon "
            "les tiers définis dans core/models.py."
        )

    st.divider()

    st.subheader("Accès au workspace")

    read_tool_names = get_read_tool_names()
    proposal_tool_names = (
        get_proposal_tool_names()
    )
    all_tool_names = get_tool_names()

    if all_tool_names:
        st.success(
            "Accès contrôlé au workspace activé"
        )

        with st.expander(
            "Outils de consultation",
            expanded=False,
        ):
            for tool_name in read_tool_names:
                st.caption(
                    f"• {tool_name}"
                )

        with st.expander(
            "Outils de proposition",
            expanded=False,
        ):
            for tool_name in proposal_tool_names:
                st.caption(
                    f"• {tool_name}"
                )

    else:
        st.warning(
            "Aucun outil de fichier n'est disponible."
        )

    st.caption(
        "L'agent peut consulter les fichiers et préparer "
        "des modifications ciblées. Toute écriture nécessite "
        "ton approbation explicite."
    )

    st.divider()

    with st.expander(
        "Instruction système",
        expanded=False,
    ):
        system_prompt = st.text_area(
            "Comportement général de l'assistant",
            value=st.session_state.system_prompt,
            height=420,
        )

        if st.button(
            "Appliquer l'instruction",
            use_container_width=True,
        ):
            st.session_state.system_prompt = (
                system_prompt.strip()
                or DEFAULT_SYSTEM_PROMPT
            )

            st.success(
                "Instruction système mise à jour."
            )

    st.divider()

    if st.button(
        "Nouvelle conversation",
        type="primary",
        use_container_width=True,
        disabled=is_api_request_pending(),
    ):
        # On conserve la conversation courante avant d'en ouvrir
        # une nouvelle, pour ne rien perdre.
        persist_current_conversation()
        reset_conversation()
        st.rerun()

    st.divider()

    st.subheader("💬 Conversations")
    st.session_state.prompt_rewrite_enabled = st.toggle(
        "✨ Reformuler ma demande",
        value=st.session_state.get("prompt_rewrite_enabled", True),
        disabled=is_api_request_pending(),
        help=(
            "Améliore automatiquement la formulation de votre message "
            "avant l'envoi à l'IA (chat et multi-agent). "
            "Désactivez pour envoyer votre texte tel quel. "
            "Désactivé pendant une requête API."
        ),
    )

    st.divider()

    conversation_summaries = (
        conversation_store.list_conversations()
    )

    if not conversation_summaries:
        st.caption(
            "Aucune conversation enregistrée pour le moment."
        )

    for summary in conversation_summaries:
        summary_id = summary["id"]
        is_current = (
            summary_id
            == st.session_state.get("conversation_id")
        )

        title = summary.get(
            "title",
            "Conversation",
        )

        message_count = summary.get(
            "message_count",
            0,
        )

        label = (
            f"{'▶ ' if is_current else ''}{title} "
            f"({message_count})"
        )

        open_column, delete_column = st.columns(
            [4, 1]
        )

        with open_column:
            if st.button(
                label,
                key=f"open_conv_{summary_id}",
                use_container_width=True,
                disabled=is_current or is_api_request_pending(),
            ):
                # Sauvegarde de la conversation courante avant
                # de charger la conversation demandée.
                persist_current_conversation()

                loaded = (
                    conversation_store.load_conversation(
                        summary_id
                    )
                )

                if loaded is not None:
                    st.session_state.conversation_id = (
                        summary_id
                    )
                    st.session_state.messages = loaded.get(
                        "messages",
                        [],
                    )
                    st.session_state.memory = loaded.get(
                        "memory",
                        empty_memory(),
                    )
                    st.rerun()

                else:
                    st.warning(
                        "Cette conversation est introuvable "
                        "ou illisible."
                    )

        with delete_column:
            if st.button(
                "🗑️",
                key=f"delete_conv_{summary_id}",
                use_container_width=True,
                help="Supprimer cette conversation",
                disabled=is_api_request_pending(),
            ):
                try:
                    conversation_store.delete_conversation(
                        summary_id
                    )

                except conversation_store.ConversationStoreError as error:
                    st.warning(
                        "Suppression impossible : "
                        f"{error}"
                    )

                else:
                    if is_current:
                        # On repart sur une conversation neuve
                        # si l'on vient de supprimer la courante.
                        reset_conversation()

                    st.rerun()

    st.divider()

    st.subheader("📝 Modifications en attente")

    display_pending_proposals()

    st.divider()

    st.caption(
        "Authentification assurée par le rôle IAM "
        "de SageMaker AI Studio."
    )


# -------------------------------------------------------------------
# Interface principale
# -------------------------------------------------------------------

# Aiguillage de mode : en multi-agent, on rend l'onglet dédié et on stoppe
# le script ici pour ne pas exécuter le corps de chat classique situé
# plus bas (aucune réindentation nécessaire).
if st.session_state.get("app_mode") == "🧑‍🤝‍🧑 Multi-agent":
    render_multi_agent()
    st.stop()

st.title("🤖 Assistant IA personnel")

st.caption(
    "Assistant Amazon Bedrock avec accès contrôlé au workspace "
    "et validation humaine des écritures."
)

# -------------------------------------------------------------------
# Affichage de l'état de la requête API en cours
# -------------------------------------------------------------------

# Si une requête API est en cours et qu'un affichage d'état est actif,
# on recrée le placeholder après un rerun pour maintenir le feedback visuel.
# Le placeholder doit être recréé à chaque rerun car st.empty() retourne
# un objet non sérialisable.
if is_api_request_pending() and st.session_state.get("api_status_active", False):
    operation = st.session_state.api_request_operation or "opération"
    model = st.session_state.api_request_model or "modèle"
    message = f"🔄 {operation.capitalize()} en cours avec {model}..."
    
    # Création du placeholder à chaque rerun
    status_placeholder = st.empty()
    status_placeholder.info(message)



# -------------------------------------------------------------------
# Message d'accueil
# -------------------------------------------------------------------

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Bonjour ! Je peux consulter les fichiers du "
            "**workspace**, analyser la copie bêta de l'application "
            "et préparer des corrections ciblées.\n\n"
            "Toute modification doit être **explicitement approuvée** "
            "avant d'être appliquée."
        )


# -------------------------------------------------------------------
# Historique
# -------------------------------------------------------------------

for message in st.session_state.messages:
    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":
            model_label = message.get(
                "model_label"
            )

            usage = message.get(
                "usage",
                {},
            )

            tool_call_count = message.get(
                "tool_call_count",
                0,
            )

            if model_label:
                st.caption(
                    f"Modèle : {model_label}"
                )

            if usage or tool_call_count:
                display_usage_information(
                    usage=usage,
                    tool_call_count=tool_call_count,
                )


# -------------------------------------------------------------------
# Saisie utilisateur et exécution de l'agent
# -------------------------------------------------------------------

_capabilities = get_model_capabilities(selected_model)

_accepted_formats = sorted(
    _capabilities["accepted_image_formats"]
    | _capabilities["accepted_document_formats"]
)

_chat_value = st.chat_input(
    "Écris ton message ou demande une modification ciblée...",
    accept_file="multiple",
    file_type=_accepted_formats if _accepted_formats else None,
)

# st.chat_input renvoie une chaîne simple ou un objet possédant
# .text et .files selon la présence de accept_file. On lit de
# manière défensive pour rester tolérant aux variations d'API.
if isinstance(_chat_value, str):
    prompt = _chat_value
    uploaded_files = []
elif _chat_value is None:
    prompt = None
    uploaded_files = []
else:
    prompt = getattr(_chat_value, "text", "") or ""
    uploaded_files = getattr(_chat_value, "files", []) or []

# Reformulation automatique (non bloquante) si l'interrupteur est actif
# et qu'un texte a ete saisi. En cas d'echec, prompt reste inchange.
# Protection contre les reruns : l'état API persiste à travers les reruns.
if prompt and st.session_state.get("prompt_rewrite_enabled", True):
    try:
        begin_api_request("reformulation", st.session_state.selected_model)
        
        # Création d'un placeholder persistant pour l'indicateur d'état
        # Le placeholder doit être recréé à chaque rerun car st.empty()
        # retourne un objet non sérialisable. On stocke un flag pour
        # indiquer qu'un affichage d'état est actif.
        if "api_status_active" not in st.session_state:
            st.session_state.api_status_active = False
        
        st.session_state.api_status_active = True
        st.session_state.api_status_message = "🔄 Reformulation de la demande..."
        
        # Création du placeholder à chaque rerun
        status_placeholder = st.empty()
        status_placeholder.info(st.session_state.api_status_message)
        
        with st.spinner("Reformulation de la demande..."):
            _rewrite = rewrite_user_prompt(prompt, mode="chat")
        if _rewrite.ok and _rewrite.text:
            prompt = _rewrite.text
        
        # Nettoyage du placeholder persistant
        if "api_status_active" in st.session_state:
            st.session_state.api_status_active = False
            st.session_state.api_status_message = None
        
        complete_api_request()
    except Exception as exc:
        # Nettoyage du placeholder persistant en cas d'erreur
        if "api_status_active" in st.session_state:
            st.session_state.api_status_active = False
            st.session_state.api_status_message = None
        
        fail_api_request(f"Reformulation échouée : {exc}")
        st.error(f"Reformulation échouée : {exc}")
        # On continue avec le prompt original
        reset_api_request()

# Un envoi est valable s'il contient du texte OU au moins un fichier.
if prompt or uploaded_files:
    attachments = []
    attachments_meta = []

    for uploaded in uploaded_files:
        raw = uploaded.getvalue()
        kind = classify_attachment(uploaded.name, _capabilities)

        if kind is None:
            st.warning(
                f"Format non supporté par le modèle : "
                f"{uploaded.name} (ignoré)."
            )
            continue

        file_format = uploaded.name.lower().split(".")[-1]

        # Normalisation : Bedrock attend "jpeg" plutôt que "jpg".
        if file_format == "jpg":
            file_format = "jpeg"

        attachments.append(
            {
                "kind": kind,
                "name": uploaded.name,
                "format": file_format,
                "data": raw,
            }
        )

        attachments_meta.append(
            {
                "kind": kind,
                "name": uploaded.name,
                "format": file_format,
                "size": len(raw),
            }
        )

    st.session_state.current_attachments = attachments

    user_message = {
        "role": "user",
        "content": prompt,
    }

    if attachments_meta:
        user_message["attachments"] = attachments_meta

    # Empêche l'ajout d'un doublon lors de la régénération :
    # si le dernier message est déjà user avec le même contenu,
    # on ne l'ajoute pas à nouveau.
    if (
        st.session_state.messages
        and st.session_state.messages[-1].get("role") == "user"
        and st.session_state.messages[-1].get("content") == prompt
        and st.session_state.messages[-1].get("attachments", []) == attachments_meta
    ):
        # Régénération : on garde le message user existant.
        pass
    else:
        st.session_state.messages.append(
            user_message
        )

    with st.chat_message("user"):
        st.markdown(prompt)

        for meta in attachments_meta:
            icon = "🖼️" if meta["kind"] == "image" else "📄"
            size_kb = meta["size"] / 1024

            st.caption(
                f"{icon} {meta['name']} "
                f"({meta['format']}, {size_kb:.1f} Ko)"
            )

    with st.chat_message("assistant"):
        try:
            model_label = available_models[
                selected_model
            ]

            # Démarrage de la requête API avec verrouillage de l'interface
            begin_api_request("analyse", model_label)
            
            # Création d'un placeholder persistant pour l'indicateur d'état
            # Le placeholder doit être recréé à chaque rerun car st.empty()
            # retourne un objet non sérialisable. On stocke un flag pour
            # indiquer qu'un affichage d'état est actif.
            if "api_status_active" not in st.session_state:
                st.session_state.api_status_active = False
            
            st.session_state.api_status_active = True
            st.session_state.api_status_message = f"🔄 {model_label} analyse la demande..."
            
            # Création du placeholder à chaque rerun
            status_placeholder = st.empty()
            status_placeholder.info(st.session_state.api_status_message)

            with st.spinner(
                f"{model_label} analyse la demande..."
            ):
                agent = load_agent(
                    selected_model
                )

                set_memory_state(st.session_state.memory)

                conversation = build_langchain_messages(
                    messages=st.session_state.messages,
                    system_prompt=(
                        st.session_state.system_prompt
                    ),
                )

                result = agent.invoke(
                    {
                        "messages": conversation,
                    },
                    config={
                        "recursion_limit": CHAT_RECURSION_LIMIT,
                    },
                )

                agent_messages = result.get(
                    "messages",
                    [],
                )

                final_message = find_last_ai_message(
                    agent_messages
                )

                if final_message is None:
                    raise RuntimeError(
                        "L'agent n'a produit aucune "
                        "réponse textuelle."
                    )

                response_text = extract_response_text(
                    final_message.content
                )

                usage = collect_usage_metadata(
                    agent_messages
                )

                tool_call_count = count_tool_calls(
                    agent_messages
                )

                tool_call_names = collect_tool_call_names(agent_messages)
                tool_results = collect_tool_results(agent_messages)
                called_proposal_tools = [
                    name for name in tool_call_names
                    if name in set(get_proposal_tool_names())
                ]
                pending_proposals = get_pending_proposals()
                # Détecte les créations RÉUSSIES à partir du résultat
                # structuré des outils (`_tool_result` renvoie un JSON
                # contenant `created: true`). C'est la source de vérité
                # fiable : l'état `pending_proposals` peut être vidé par
                # un rerun ou une persistance alors qu'une proposition a
                # bien été créée. On évite ainsi un faux avertissement.
                proposal_created_ok = False
                for item in tool_results:
                    content = item.get("content", "")
                    try:
                        payload = json.loads(content)
                    except (TypeError, ValueError):
                        payload = None
                    if isinstance(payload, dict):
                        # Résultat structuré : `created` fait foi. Ne pas
                        # retomber sur la détection textuelle, qui confondrait
                        # « Aucune proposition créée » avec un succès.
                        if payload.get("created"):
                            proposal_created_ok = True
                            break
                        continue
                    if "proposition créée" in content.casefold():
                        proposal_created_ok = True
                        break
            
            # Nettoyage de l'indicateur d'état persistant
            if "api_status_active" in st.session_state:
                st.session_state.api_status_active = False
                st.session_state.api_status_message = None
            
            # Fin de la requête API avec succès
            complete_api_request()
            
            st.markdown(response_text)

            st.caption(
                f"Modèle : {model_label}"
            )

            display_usage_information(
                usage=usage,
                tool_call_count=tool_call_count,
            )

            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "model_alias": selected_model,
                "model_label": model_label,
                "usage": usage,
                "tool_call_count": tool_call_count,
            }

            st.session_state.messages.append(
                assistant_message
            )

            # Purge des octets du tour courant : ils ne doivent pas
            # être réinjectés au tour suivant (économie de tokens
            # et cohérence de l'historique dégradé).
            st.session_state.current_attachments = []

            # Relit la mémoire éventuellement mise à jour par l'agent
            # (via l'outil `mettre_a_jour_memoire`) pour la refléter
            # dans la session avant la persistance.
            st.session_state.memory = get_memory_state()

            # Persistance immédiate de la conversation sur disque
            # (mémoire glissante + messages) après chaque réponse.
            persist_current_conversation()
            proposal_refusals = [
                item for item in tool_results
                if "proposition refusée" in item["content"].casefold()
                or "aucune proposition créée" in item["content"].casefold()
            ]

            if proposal_refusals:
                with st.expander("Propositions refusées", expanded=True):
                    for refusal in proposal_refusals:
                        st.error(refusal["content"])
            elif called_proposal_tools and not pending_proposals and not proposal_created_ok:
                st.warning(
                    "Un outil de proposition a été appelé, mais aucune "
                    "proposition en attente n'a été trouvée. Consulte les "
                    "résultats techniques des outils."
                )
            if pending_proposals:
                st.rerun()

        except GraphRecursionError as error:
            # Nettoyage du placeholder persistant en cas d'erreur
            if "api_status_active" in st.session_state:
                st.session_state.api_status_active = False
                st.session_state.api_status_message = None
            
            fail_api_request("Limite d'étapes atteinte")
            st.error(
                "L'agent a atteint la limite d'étapes sans produire de "
                "condition d'arrêt. Un outil peut avoir été appelé en boucle."
            )
            with st.expander("Détails techniques", expanded=False):
                st.exception(error)
            pending_after_error = get_pending_proposals()
            if pending_after_error:
                st.warning(
                    f"{len(pending_after_error)} proposition(s) créées avant "
                    "l'interruption restent disponibles dans la barre latérale."
                )
        except Exception as error:
            # Nettoyage du placeholder persistant en cas d'erreur
            if "api_status_active" in st.session_state:
                st.session_state.api_status_active = False
                st.session_state.api_status_message = None
            
            fail_api_request(str(error))
            st.error(
                "Une erreur est survenue pendant "
                "l'exécution de l'agent."
            )

            with st.expander(
                "Détails techniques",
                expanded=False,
            ):
                st.exception(error)
            pending_after_error = get_pending_proposals()
            if pending_after_error:
                st.warning(
                    f"{len(pending_after_error)} proposition(s) ont été créées "
                    "avant l'interruption. Elles sont conservées dans la barre latérale."
                )
                if st.button(
                    "Afficher les propositions en attente",
                    key="show_pending_proposals_after_error",
                ):
                    st.rerun()