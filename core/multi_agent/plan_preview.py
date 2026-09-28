"""Composant Streamlit de prévisualisation et édition de plan multi-agent.

Ce module fournit une interface interactive pour visualiser et modifier un plan
avant son exécution. Il consomme la structure réelle du TaskLedger et affiche :
- La liste des workers avec leurs tâches assignées
- Les connexions de communication (dépendances + chef→workers)
- Les prompts système et de tâche de chaque agent
- Des champs éditables pour modifier le plan

Le composant retourne un plan édité dans la même structure que le plan d'entrée.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from core.models import get_tier, MODEL_TIERS
from core.multi_agent.ledger import Task, TaskLedger
from core.multi_agent.roles import (
    ASSIGNABLE_TIERS,
    worker_system_prompt,
    worker_task_prompt,
)
from core.multi_agent.workers import build_context


def _enrich_description(
    description: str,
    expected_output: str,
    criteria: list[str],
    requires_review: bool,
) -> str:
    """Construit le format canonique consommé par l'exécution des workers."""
    parts = [description.strip()]
    if expected_output.strip():
        parts.append(f"Sortie attendue : {expected_output.strip()}")
    if criteria:
        cleaned = [criterion.strip() for criterion in criteria if criterion.strip()]
        if cleaned:
            parts.append(
                "Critères d'acceptation :\n"
                + "\n".join(f"- {criterion}" for criterion in cleaned)
            )
    if requires_review:
        parts.append("Revue humaine ou critique ciblée requise avant synthèse.")
    return "\n\n".join(part for part in parts if part)


def _validate_dependencies(
    ledger: TaskLedger,
    edited_tasks: dict[str, dict[str, Any]],
) -> str | None:
    """Valide que les dépendances éditées forment un graphe acyclique."""
    task_ids = {task.id for task in ledger.tasks}
    graph: dict[str, list[str]] = {}
    for task in ledger.tasks:
        dependencies = list(
            edited_tasks.get(task.id, {}).get("dependencies", task.dependencies)
        )
        if task.id in dependencies:
            return f"{task.id} ne peut pas dépendre de lui-même."
        unknown = sorted(set(dependencies) - task_ids)
        if unknown:
            return f"{task.id} dépend de tâche(s) inconnue(s) : {', '.join(unknown)}."
        graph[task.id] = dependencies

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> str | None:
        if task_id in visiting:
            return f"Cycle de dépendances détecté autour de {task_id}."
        if task_id in visited:
            return None
        visiting.add(task_id)
        for dependency in graph[task_id]:
            error = visit(dependency)
            if error:
                return error
        visiting.remove(task_id)
        visited.add(task_id)
        return None

    for task_id in graph:
        error = visit(task_id)
        if error:
            return error
    return None


def render_plan_preview_editor(ledger: TaskLedger, discovery_context: str = "") -> TaskLedger | None:
    """Affiche un plan éditable et retourne le ledger modifié si validé.

    Args:
        ledger: Le TaskLedger contenant le plan à prévisualiser/éditer
        discovery_context: Contexte de découverte pour construire les prompts

    Returns:
        TaskLedger modifié si l'utilisateur valide les changements, None sinon

    Points d'appel réels :
    - worker_system_prompt() : génère le prompt système pour chaque tier
    - worker_task_prompt() : génère le prompt de tâche avec contexte
    - get_tier() : récupère la configuration du tier (label, modèle, etc.)
    - Task.dependencies : liste des IDs de tâches dépendantes
    """
    st.markdown("### 📋 Prévisualisation et édition du plan")
    
    # Section 1 : Vue d'ensemble de l'équipe
    _render_team_overview(ledger)
    
    # Section 2 : Graphe des connexions
    _render_connections_graph(ledger)
    
    # Section 3 : Édition détaillée des tâches
    edited_tasks = _render_task_editor(ledger, discovery_context)
    
    # Section 4 : Actions de validation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Valider les modifications", type="primary", use_container_width=True):
            # Appliquer les modifications au ledger
            try:
                return _apply_edits(ledger, edited_tasks)
            except ValueError as exc:
                st.error(str(exc))
    with col2:
        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.rerun()
    
    return None


def _render_team_overview(ledger: TaskLedger) -> None:
    """Affiche la vue d'ensemble de l'équipe et la répartition des tâches."""
    st.markdown("#### 👥 Équipe mobilisée")
    
    # Statistiques globales
    total_tasks = len(ledger.tasks)
    team_structure = ledger.team_structure()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tâches totales", total_tasks)
    with col2:
        st.metric("Tiers utilisés", len(team_structure))
    with col3:
        review_count = sum(1 for task in ledger.tasks if task.requires_review)
        st.metric("Tâches à revoir", review_count)
    
    # Répartition par tier
    st.markdown("**Répartition par tier :**")
    for tier, tasks in sorted(team_structure.items()):
        tier_config = get_tier(tier)
        tier_label = tier_config["label"]
        model_alias = tier_config["model_alias"]
        st.caption(
            f"- **{tier_label}** ({model_alias}) : {len(tasks)} agent(s) — "
            f"{', '.join(task.id for task in tasks)}"
        )


def _render_connections_graph(ledger: TaskLedger) -> None:
    """Affiche le graphe des connexions entre agents.
    
    Visualise :
    - Les dépendances entre workers (depends_on)
    - Le lien implicite chef → tous les workers
    """
    st.markdown("#### 🔗 Graphe des connexions")
    
    with st.expander("Voir le graphe de communication", expanded=False):
        st.caption("**Chef de projet → Workers**")
        st.markdown(
            "Le Chef coordonne tous les workers. Il ne communique jamais "
            "directement avec les workers pendant l'exécution : la communication "
            "passe par le ledger (source de vérité)."
        )
        
        st.caption("**Dépendances entre workers**")
        has_dependencies = False
        for task in ledger.tasks:
            if task.dependencies:
                has_dependencies = True
                deps_str = ", ".join(task.dependencies)
                st.markdown(f"- **{task.id}** ({task.title}) dépend de : {deps_str}")
        
        if not has_dependencies:
            st.caption("Aucune dépendance entre workers. Toutes les tâches sont parallélisables.")
        
        # Visualisation textuelle du graphe
        st.caption("**Graphe complet (format texte) :**")
        graph_lines = ["```"]
        graph_lines.append("Chef de projet")
        for task in ledger.tasks:
            prefix = "  ├─" if task != ledger.tasks[-1] else "  └─"
            graph_lines.append(f"{prefix} {task.id} ({task.assigned_tier})")
            if task.dependencies:
                for dep in task.dependencies:
                    graph_lines.append(f"      ↳ dépend de {dep}")
        graph_lines.append("```")
        st.markdown("\n".join(graph_lines))


def _render_task_editor(ledger: TaskLedger, discovery_context: str) -> dict[str, dict[str, Any]]:
    """Affiche l'éditeur détaillé de chaque tâche avec prompts.
    
    Retourne un dictionnaire {task_id: {champ: nouvelle_valeur}} des modifications.
    
    Points d'appel réels :
    - worker_system_prompt(tier_label) : génère le prompt système
    - worker_task_prompt(title, description, context, criteria, output) : génère le prompt de tâche
    - build_context() est simulé ici car il nécessite le ledger complet
    """
    st.markdown("#### ✏️ Édition des tâches et prompts")
    
    edited_tasks: dict[str, dict[str, Any]] = {}
    
    for task in ledger.tasks:
        with st.expander(f"**{task.id}** — {task.title}", expanded=False):
            # Clé unique pour chaque widget
            key_prefix = f"edit_{task.id}"
            
            # Édition des métadonnées de la tâche
            st.markdown("##### Métadonnées de la tâche")
            
            col1, col2 = st.columns(2)
            with col1:
                new_title = st.text_input(
                    "Titre",
                    value=task.title,
                    key=f"{key_prefix}_title",
                    help="Titre court orienté résultat"
                )
            with col2:
                new_tier = st.selectbox(
                    "Tier assigné",
                    options=ASSIGNABLE_TIERS,
                    index=ASSIGNABLE_TIERS.index(task.assigned_tier) if task.assigned_tier in ASSIGNABLE_TIERS else 1,
                    key=f"{key_prefix}_tier",
                    help="Tier du worker (nano/fast/smart/expert)"
                )

            new_capabilities = st.multiselect(
                "Capacités déclarées",
                options=["auto", "read", "propose_write", "generate"],
                default=list(getattr(task, "capabilities", []) or ["auto"]),
                key=f"{key_prefix}_capabilities",
                help="Mode d'outil autorisé : lecture, proposition d'écriture ou génération.",
            )
            budget_col1, budget_col2 = st.columns(2)
            with budget_col1:
                new_max_retries = st.number_input(
                    "Retries maximum",
                    min_value=0,
                    max_value=2,
                    value=int(getattr(task, "max_retries", 1)),
                    key=f"{key_prefix}_max_retries",
                )
            with budget_col2:
                new_max_escalations = st.number_input(
                    "Escalades maximum",
                    min_value=0,
                    max_value=1,
                    value=int(getattr(task, "max_escalations", 1)),
                    key=f"{key_prefix}_max_escalations",
                )
            
            # Description enrichie (extraire les parties)
            description_parts = task.description.split("\n\nSortie attendue : ")
            base_description = description_parts[0]
            
            # Extraire expected_output et criteria si présents
            expected_output = ""
            criteria = []
            if len(description_parts) > 1:
                remaining = description_parts[1]
                output_parts = remaining.split("\n\nCritères d'acceptation :\n")
                expected_output = output_parts[0]
                if len(output_parts) > 1:
                    criteria_text = output_parts[1]
                    # Retirer la ligne de revue si présente
                    criteria_text = criteria_text.split("\n\nRevue humaine")[0]
                    criteria = [
                        line.strip("- ").strip()
                        for line in criteria_text.split("\n")
                        if line.strip().startswith("- ")
                    ]
            
            new_description = st.text_area(
                "Description",
                value=base_description,
                key=f"{key_prefix}_description",
                height=100,
                help="Consigne autonome et bornée"
            )
            
            new_expected_output = st.text_input(
                "Sortie attendue",
                value=expected_output,
                key=f"{key_prefix}_output",
                help="Forme attendue du livrable"
            )
            
            new_criteria_text = st.text_area(
                "Critères d'acceptation (un par ligne)",
                value="\n".join(criteria),
                key=f"{key_prefix}_criteria",
                height=80,
                help="Critères observables, un par ligne"
            )
            new_criteria = [
                line.strip() for line in new_criteria_text.split("\n")
                if line.strip()
            ]
            
            new_requires_review = st.checkbox(
                "Requiert une revue critique",
                value=task.requires_review,
                key=f"{key_prefix}_review",
                help="Si activé, le livrable sera soumis à un agent critique indépendant"
            )
            
            # Dépendances
            available_deps = [t.id for t in ledger.tasks if t.id != task.id]
            current_deps = task.dependencies if task.dependencies else []
            new_dependencies = st.multiselect(
                "Dépendances (depends_on)",
                options=available_deps,
                default=[dep for dep in current_deps if dep in available_deps],
                key=f"{key_prefix}_deps",
                help="Tâches dont ce worker dépend (doit attendre leur livrable)"
            )
            dependency_states = ["completed", "applied", "tested", "verified"]
            current_dependency_state = getattr(task, "dependency_state", "completed")
            new_dependency_state = st.selectbox(
                "État requis des dépendances",
                options=dependency_states,
                index=dependency_states.index(current_dependency_state)
                if current_dependency_state in dependency_states else 0,
                key=f"{key_prefix}_dependency_state",
                help="completed utilise le résultat ; applied/tested/verified attend cet état réel.",
            )
            
            # Affichage des prompts réels
            st.markdown("##### 🤖 Prompts générés (lecture seule)")
            
            # Prompt système : appel réel à worker_system_prompt()
            tier_config = get_tier(new_tier)
            tier_label = tier_config["label"]
            system_prompt = worker_system_prompt(tier_label)
            
            with st.expander("Prompt système", expanded=False):
                st.code(system_prompt, language="text")
                st.caption(
                    f"Généré par `worker_system_prompt('{tier_label}')` "
                    f"depuis core/multi_agent/roles.py"
                )
            
            # Prompt de tâche : appel réel à worker_task_prompt()
            # Construction du contexte (simplifié ici, la vraie construction
            # se fait dans workers.build_context())
            context_parts = []
            if discovery_context:
                context_parts.append(
                    "CONTEXTE DE DECOUVERTE DU WORKSPACE (extraits PARTIELS et TRONQUES, "
                    "lus avant planification).\n\n"
                    "[... contexte de découverte ...]"
                )
            
            # Ajouter les livrables des dépendances (simulé)
            if new_dependencies:
                dep_deliverables = []
                for dep_id in new_dependencies:
                    dep_task = ledger.get_task(dep_id)
                    if dep_task:
                        dep_deliverables.append(
                            f"### {dep_task.title}\n"
                            f"[... livrable de {dep_id} ...]"
                        )
                if dep_deliverables:
                    context_parts.append(
                        "CONTEXTE NON AUTORITATIF FOURNI PAR D'AUTRES WORKERS.\n\n"
                        + "\n\n".join(dep_deliverables)
                    )
            
            preview_task = Task(
                id=task.id,
                title=new_title,
                description=new_description,
                assigned_tier=new_tier,
                dependencies=new_dependencies,
                dependency_state=new_dependency_state,
            )
            context = build_context(ledger, preview_task, discovery_context)
            enriched_description = _enrich_description(
                new_description,
                new_expected_output,
                new_criteria,
                new_requires_review,
            )
            task_prompt = worker_task_prompt(
                title=new_title,
                description=enriched_description,
                context=context,
                target_output_tokens=1200,
            )
            if discovery_context or new_dependencies:
                st.info(
                    "Prévisualisation fidèle du format du prompt, mais le contexte "
                    "est partiel : les livrables des dépendances seront injectés "
                    "uniquement après leur exécution."
                )
            
            with st.expander("Prompt de tâche", expanded=False):
                st.code(task_prompt, language="text")
                st.caption(
                    "Généré par `worker_task_prompt(title, description, context)` "
                    "comme lors de l'exécution."
                )
            
            # Stocker les modifications
            edited_tasks[task.id] = {
                "title": new_title,
                "description": new_description,
                "expected_output": new_expected_output,
                "criteria": new_criteria,
                "tier": new_tier,
                "capabilities": new_capabilities or ["auto"],
                "max_retries": int(new_max_retries),
                "max_escalations": int(new_max_escalations),
                "requires_review": new_requires_review,
                "dependencies": new_dependencies,
                "dependency_state": new_dependency_state,
            }
    
    return edited_tasks


def _apply_edits(ledger: TaskLedger, edited_tasks: dict[str, dict[str, Any]]) -> TaskLedger:
    """Applique les modifications au ledger et retourne le ledger modifié.
    
    Reconstruit la description enrichie au format attendu par le système.
    """
    dependency_error = _validate_dependencies(ledger, edited_tasks)
    if dependency_error:
        raise ValueError(dependency_error)

    for task in ledger.tasks:
        edits = edited_tasks.get(task.id)
        if edits is None:
            continue
        capabilities = list(edits.get("capabilities", task.capabilities or ["auto"]))
        if not capabilities or not set(capabilities).issubset(
            {"auto", "read", "propose_write", "generate"}
        ):
            raise ValueError(f"{task.id} contient une capacité inconnue.")
        max_retries = int(edits.get("max_retries", task.max_retries))
        max_escalations = int(edits.get("max_escalations", task.max_escalations))
        if not 0 <= max_retries <= 2 or not 0 <= max_escalations <= 1:
            raise ValueError(f"{task.id} contient un budget hors limites.")
        title = str(edits.get("title", task.title))
        description = str(edits.get("description", task.description))
        if not title.strip():
            raise ValueError(f"{task.id} doit conserver un titre non vide.")
        if not description.strip():
            raise ValueError(f"{task.id} doit conserver une description non vide.")
        edits["title"] = title
        edits["description"] = description
        edits["capabilities"] = capabilities
        edits["max_retries"] = max_retries
        edits["max_escalations"] = max_escalations

    for task in ledger.tasks:
        if task.id not in edited_tasks:
            continue
        
        edits = edited_tasks[task.id]

        # Mettre à jour les champs simples
        task.title = edits["title"].strip()
        task.assigned_tier = edits.get("tier", task.assigned_tier)
        task.capabilities = list(edits["capabilities"])
        task.max_retries = edits["max_retries"]
        task.max_escalations = edits["max_escalations"]
        task.requires_review = edits.get("requires_review", task.requires_review)
        task.dependencies = edits.get("dependencies", task.dependencies)
        dependency_state = edits.get("dependency_state", task.dependency_state)
        if dependency_state not in {"completed", "applied", "tested", "verified"}:
            raise ValueError(f"{task.id} contient un état de dépendance inconnu.")
        task.dependency_state = dependency_state
        
        task.description = _enrich_description(
            edits["description"],
            edits.get("expected_output", ""),
            edits.get("criteria", []),
            edits.get("requires_review", task.requires_review),
        )
    
    return ledger


# Point d'intégration dans ui.py
# 
# Pour intégrer ce composant dans l'interface multi-agent existante,
# ajouter dans core/multi_agent/ui.py, fonction _phase_planif() :
#
# ```python
# from core.multi_agent.plan_preview import render_plan_preview_editor
#
# def _phase_planif(chief) -> None:
#     report = st.session_state.get(_KEY_PLAN_REPORT)
#     st.subheader("Plan propose")
#     
#     # ... code existant (affichage du rapport) ...
#     
#     # NOUVEAU : Composant de prévisualisation éditable
#     with st.expander("🔍 Prévisualiser et éditer le plan", expanded=False):
#         discovery_ctx = getattr(chief.discovery_report, "context_summary", "") if chief.discovery_report else ""
#         edited_ledger = render_plan_preview_editor(chief.ledger, discovery_ctx)
#         if edited_ledger is not None:
#             chief.ledger = edited_ledger
#             st.rerun()
#     
#     # ... reste du code existant ...
# ```
#
# Ce point d'intégration :
# - S'insère naturellement dans la phase PLANIF existante
# - Utilise le ledger réel du chef (chief.ledger)
# - Récupère le contexte de découverte réel (chief.discovery_report)
# - Applique les modifications au ledger du chef si validées
# - Déclenche un rerun pour rafraîchir l'affichage
