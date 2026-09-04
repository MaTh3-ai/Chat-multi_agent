# Plan de l'application (mémo interne)

> Fichier de référence rapide. Mis à jour à chaque évolution structurelle.
> Objectif : retrouver les points d'intégration sans relire tout le code.

## Arborescence

```
documents/
├── app.py                  # Interface Streamlit (UI + orchestration)
├── llm_engine.py           # Création modèle/agent Bedrock + registre d'outils
├── ARCHITECTURE.md         # Ce fichier
├── core/
│   ├── models.py           # Catalogue des modèles Bedrock
│   ├── conversation_store.py  # Persistance disque des conversations (JSON)
│   └── memory_tools.py     # Outil `mettre_a_jour_memoire` (mémoire agent)
└── tools/
    ├── custom_tools.py     # Outils lecture workspace + PROJECT_ROOT / WORKSPACE_ROOT
    └── write_tools.py      # Outils de proposition/écriture (approbation humaine)
```

## Racines de chemins (définies dans tools/custom_tools.py)
- `PROJECT_ROOT` : racine projet (parent de `workspace/`).
- `WORKSPACE_ROOT` : `PROJECT_ROOT / "workspace"` (zone accessible aux outils).
- `backups/` et `conversations/` vivent sous `PROJECT_ROOT` (hors workspace).

## Flux d'un tour de conversation (app.py, bas du fichier)
1. `st.chat_input` -> append message user dans `st.session_state.messages`.
2. `build_langchain_messages(messages, system_prompt)` construit les messages LangChain.
   - Injecte le bloc `[MÉMOIRE ...]` dans le SystemMessage.
   - Applique une fenêtre glissante (N derniers messages) pour limiter les tokens.
3. `agent.invoke({"messages": conversation})`.
4. Extraction réponse + usage + tool calls -> append message assistant.
5. Sauvegarde auto sur disque via conversation_store.

## Structure d'un message (session_state.messages)
- `role`: "user" | "assistant"
- `content`: str
- assistant uniquement (non persistés) : `model_alias`, `model_label`, `usage`, `tool_call_count`
- **Persistés** : uniquement `role`, `content`, `model_label`.

## Persistance (core/conversation_store.py)
- Un fichier JSON par conversation dans `PROJECT_ROOT/conversations/`.
- Schéma fichier :
  ```json
  {
    "id": "uuid",
    "title": "…",
    "created_at": "iso",
    "updated_at": "iso",
    "messages": [{"role", "content", "model_label"}],
    "memory": {"summary": "", "tasks": [], "decisions": [], "files": []}
  }
  ```
- Limites : `MAX_CONVERSATIONS` (nb de conversations), `MAX_MESSAGES_PER_CONV`.
- Écriture atomique (fichier temporaire + os.replace).

## Mémoire agent (core/memory_tools.py)
- Approche (b) : outil `mettre_a_jour_memoire(summary, tasks, decisions, files)`
  que l'agent appelle lui-même quand un fait durable doit être retenu.
- Coût quasi nul (pas d'appel LLM séparé). Bloc borné en taille.
- Le bloc mémoire est rendu texte via `format_memory_block()` et injecté
  dans le SystemMessage par `build_langchain_messages`.

## Agent / outils (llm_engine.py)
- `TOOLS = READ_TOOLS + PROPOSAL_TOOLS`.
- `get_bedrock_agent(..., additional_tools=None)` : ajoute des outils au vol.
- Les outils d'approbation/refus ne sont JAMAIS donnés au modèle.

## Approbation des propositions (write_tools.approve_proposal)
- Si le fichier n'a pas changé depuis la création : on écrit `new_content`.
- Sinon, pour un patch ciblé (`replace` / `insert` / `delete`), on
  **réapplique** l'opération sur le contenu actuel via
  `_reapply_proposal_on_current_content`. Cela permet d'approuver
  successivement plusieurs propositions ciblées sur un même fichier.
- On ne lève `ConcurrentModificationError` que si l'ancre/texte cible a
  disparu ou est devenu ambigu, ou pour une réécriture complète
  (`full_content`) sur un fichier modifié entre-temps.
- L'UI propose un bouton « Tout approuver » (sidebar).

## UI des propositions (app.py)
- Les propositions en attente sont affichées dans la **barre latérale**
  (section « 📝 Modifications en attente »), pas dans le flux du chat.

## Mode multi-agent

### Activation
- **Sélecteur** dans la sidebar (`app.py`, ligne ~1128) : `st.radio` « Mode » avec options `💬 Chat` / `🧑‍🤝‍🧑 Multi-agent`.
- La valeur choisie est stockée dans `st.session_state.app_mode`.
- Si `app_mode == "🧑‍🤝‍🧑 Multi-agent"`, `app.py` appelle `render_multi_agent()` (depuis `core.multi_agent.ui`) puis `st.stop()` pour court-circuiter le flux chat classique.

### Architecture
- **Orchestrator-Worker hiérarchique** : un **Chef de projet** (tier `expert`, seul interlocuteur de l'utilisateur) coordonne des **Workers** (tiers `nano`, `fast`, `smart` selon la complexité de chaque tâche).
- **Communication en étoile** via un **TASK LEDGER** partagé (`core/multi_agent/ledger.py`) : chaque worker lit sa tâche, écrit son résultat, le Chef agrège.
- **Human-in-the-loop** : validation utilisateur obligatoire entre chaque phase (plan → exec, exec → synthèse).

### Boucle d'états (orchestrator.py)
1. **INTERVIEW** : le Chef cadre l'objectif (questions/réponses itératives).
2. **PLANIF** : décomposition en tâches + assignation de tier → affichage visuel de l'équipe (👑 Chef + panneau par tier/modèle mobilisé) → **validation humaine**.
3. **EXEC** : workers en parallèle (via `asyncio.gather`), chacun lit/écrit dans le ledger.
4. **SYNTHESE** : le Chef agrège les résultats → **retours utilisateur** (itération possible).

### TASK LEDGER (ledger.py)
- Classe `TaskLedger` : `tasks: dict[task_id, TaskEntry]`.
- `TaskEntry` : `task_id`, `description`, `assigned_tier`, `status` (`pending`/`in_progress`/`done`/`failed`), `result`, `error`.
- Méthodes : `add_task`, `get_task`, `update_status`, `set_result`, `set_error`, `get_all_tasks`, `get_tasks_by_status`.

### Tiers de modèles (models.py)
- `MODEL_TIERS` : `nano` (nova-micro), `fast` (claude-haiku-4.5), `smart` (claude-sonnet-4.5), `expert` (claude-opus-4.8).
- `CHIEF_TIER = "expert"` : le Chef utilise toujours le modèle le plus puissant.
- `ASSIGNABLE_TIERS = ["nano", "fast", "smart"]` : tiers disponibles pour les workers.
- `get_tier(model_alias)` : retourne le tier d'un modèle.
- `resolve_tier_to_model_alias(tier)` : retourne l'alias du modèle pour un tier donné.

### Prompts (roles.py)
- `ChefDeProjet` : prompt système détaillé (cadrage, décomposition, synthèse, ton professionnel).
- `Worker` : prompt système générique (exécution autonome d'une tâche, rapport structuré).

### Visuel d'équipe (ui.py)
- Fonction `_render_team_structure(plan)` : affiche une carte `👑 **Chef de projet**` + un panneau par tier mobilisé (avec `resolve_tier_to_model_alias` pour afficher le modèle réel).
- Tableau de bord des tâches : statut, tier assigné, résultat/erreur.
- Journal live : expanders par tâche avec contenu détaillé.

### Socle LLM
- `llm_engine.create_chat_model(model_alias)` : utilisé par le Chef et chaque worker pour instancier son modèle.
- Pas d'agent LangGraph dans ce mode : appels directs `model.invoke([messages])`.

## Invariants à respecter
- Aucune écriture fichier sans approbation humaine (write_tools).
- Ne jamais persister les métadonnées lourdes de token (usage/tool_call_count).
- Le workspace peut être en lecture seule dans l'environnement d'édition.
