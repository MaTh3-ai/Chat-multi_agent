"""Interface Streamlit du mode multi-agent.

Rend l'onglet dedie quand app_mode vaut Multi-agent. Pilote la boucle
d'etats de l'orchestrateur avec validation humaine entre chaque phase :
INTERVIEW -> PLANIF -> [validation] -> EXEC -> SYNTHESE -> [retours].

L'etat du chantier est isole sous des cles st.session_state prefixees
"ma_" pour ne jamais interferer avec le flux de chat classique. La
reformulation (core.prompt_rewriter) est branchee a l'entree de
l'interview, coupable via st.session_state.prompt_rewrite_enabled.
"""

from __future__ import annotations

import streamlit as st

from core.models import resolve_tier_to_model_alias
from core.prompt_rewriter import rewrite_user_prompt

from core.multi_agent.ledger import (
    STATE_INTERVIEW,
    STATE_PLANIF,
    STATE_EXEC,
    STATE_SYNTHESE,
    STATUS_TODO,
    STATUS_DOING,
    STATUS_DONE,
    STATUS_REVIEW,
)
from core.multi_agent.orchestrator import new_chief


_KEY_CHIEF = "ma_chief"
_KEY_PLAN_REPORT = "ma_plan_report"
_KEY_LAST_REPLY = "ma_last_reply"

_STATUS_ICON = {
    STATUS_TODO: "attente",
    STATUS_DOING: "en cours",
    STATUS_DONE: "ok",
    STATUS_REVIEW: "a revoir",
}


def _get_chief():
    """Retourne l'orchestrateur courant, cree au premier acces."""
    if _KEY_CHIEF not in st.session_state or st.session_state[_KEY_CHIEF] is None:
        _reset_chief()
    return st.session_state[_KEY_CHIEF]


def _reset_chief() -> None:
    """Reinitialise completement le chantier multi-agent."""
    st.session_state[_KEY_CHIEF] = new_chief()
    st.session_state[_KEY_PLAN_REPORT] = None
    st.session_state[_KEY_LAST_REPLY] = None


def render_multi_agent() -> None:
    """Point d'entree : rend l'onglet multi-agent selon l'etat courant."""
    st.title("Mode multi-agent")
    st.caption(
        "Un Chef de projet cadre votre besoin, forme une equipe de modeles "
        "et coordonne leur travail. Vous validez chaque etape."
    )

    if st.button("Nouveau chantier"):
        _reset_chief()
        st.rerun()

    chief = _get_chief()
    state = chief.ledger.state

    if state == STATE_INTERVIEW:
        _phase_interview(chief)
    elif state == STATE_PLANIF:
        _phase_planif(chief)
    elif state == STATE_EXEC:
        _phase_exec(chief)
    elif state == STATE_SYNTHESE:
        _phase_synthese(chief)
    else:
        _render_task_board(chief)

    _render_events(chief)


def _phase_interview(chief) -> None:
    """Phase INTERVIEW : dialogue de cadrage puis passage au plan."""
    st.info(
        "Decrivez votre besoin. Le Chef de projet pose des questions pour "
        "cadrer l'objectif avant de proposer un plan."
    )
    for role, content in chief.history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    message = st.chat_input("Decrivez ou precisez votre besoin...")
    if message:
        text = message
        if st.session_state.get("prompt_rewrite_enabled", True):
            rewrite = rewrite_user_prompt(text, mode="agent")
            if rewrite.ok and rewrite.text:
                text = rewrite.text
        with st.spinner("Le Chef de projet reflechit..."):
            chief.interview(text)
        st.rerun()

    if chief.history and st.button("Le besoin est clair, proposer un plan", type="primary"):
        with st.spinner("Elaboration du plan..."):
            try:
                st.session_state[_KEY_PLAN_REPORT] = chief.plan()
            except Exception as error:
                st.error("Echec de la planification.")
                with st.expander("Details techniques", expanded=False):
                    st.exception(error)
                return
        st.rerun()


def _phase_planif(chief) -> None:
    """Phase PLANIF : plan + validation humaine avant execution."""
    report = st.session_state.get(_KEY_PLAN_REPORT)
    st.subheader("Plan propose")
    if report is not None:
        st.caption(f"{report.task_count} tache(s) - repartition : {report.tiers_used}")
        for correction in report.corrections:
            st.caption(f"Ajustement automatique : {correction}")

    _render_team_structure(chief)
    _render_task_board(chief)

    left, right = st.columns(2)
    with left:
        if st.button("Valider le plan et lancer l'execution", type="primary"):
            with st.spinner("Execution des taches par l'equipe..."):
                try:
                    chief.execute()
                    st.session_state[_KEY_LAST_REPLY] = chief.synthesize()
                except Exception as error:
                    st.error("Echec de l'execution.")
                    with st.expander("Details techniques", expanded=False):
                        st.exception(error)
                    return
            st.rerun()
    with right:
        if st.button("Revenir a l'interview"):
            chief.ledger.state = STATE_INTERVIEW
            st.rerun()


def _phase_exec(chief) -> None:
    """Phase EXEC : execution progressive des vagues avec feedback temps reel."""
    st.subheader("Execution en cours")
    _render_team_structure(chief)
    _render_task_board(chief)
    
    with st.spinner("L'equipe travaille sur la vague en cours..."):
        has_more = chief.execute_one_wave()
    
    if has_more:
        st.rerun()
    else:
        with st.spinner("Synthese finale..."):
            st.session_state[_KEY_LAST_REPLY] = chief.synthesize()
        st.rerun()


def _phase_synthese(chief) -> None:
    """Phase SYNTHESE : reponse finale + retours utilisateur (iteration)."""
    st.subheader("Reponse finale")
    reply = st.session_state.get(_KEY_LAST_REPLY) or chief.ledger.final_answer
    if reply:
        st.markdown(reply)

    with st.expander("Detail du travail de l'equipe", expanded=False):
        _render_team_structure(chief)
        _render_task_board(chief)

    feedback = st.chat_input("Un retour ? Le Chef reverra le plan en consequence.")
    if feedback:
        chief.apply_feedback(feedback)
        with st.spinner("Nouveau plan tenant compte de votre retour..."):
            try:
                st.session_state[_KEY_PLAN_REPORT] = chief.plan()
            except Exception as error:
                st.error("Echec de la re-planification.")
                with st.expander("Details techniques", expanded=False):
                    st.exception(error)
                return
        st.rerun()


def _render_team_structure(chief) -> None:
    """Affiche le Chef puis un panneau par tier mobilise."""
    st.markdown("#### Equipe mobilisee")
    st.markdown("Chef de projet : coordination, plan et synthese.")
    grouped = chief.ledger.team_structure()
    if not grouped:
        st.caption("Aucune tache assignee pour le moment.")
        return
    columns = st.columns(len(grouped))
    for column, (tier, tasks) in zip(columns, grouped.items()):
        with column:
            try:
                model_alias = resolve_tier_to_model_alias(tier)
            except Exception:
                model_alias = "?"
            st.markdown(f"**Tier {tier}**")
            st.caption(f"Modele : {model_alias}")
            st.caption(f"{len(tasks)} tache(s)")


def _render_task_board(chief) -> None:
    """Liste les taches du ledger avec statut, tier et dependances."""
    tasks = chief.ledger.tasks
    if not tasks:
        return
    st.markdown("#### Tableau des taches")
    for task in tasks:
        label = _STATUS_ICON.get(task.status, task.status)
        header = f"[{label}] {task.title} - {task.assigned_tier}"
        with st.expander(header, expanded=False):
            if task.description:
                st.caption(task.description)
            if task.dependencies:
                st.caption("Depend de : " + ", ".join(task.dependencies))
            if task.deliverable:
                st.markdown(task.deliverable)


def _render_events(chief) -> None:
    """Affiche le journal d'execution de l'orchestrateur."""
    if not chief.events:
        return
    with st.expander("Journal d'execution", expanded=False):
        for event in chief.events:
            st.caption(f"[{event.kind}] {event.message}")
