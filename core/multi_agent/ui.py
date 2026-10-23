"""Interface Streamlit du mode multi-agent.

Flux explicite : INTERVIEW -> PLANIF -> validation -> EXEC -> REVIEW ->
validation -> SYNTHESE. L'interface n'effectue aucune transition implicite
vers la synthese et expose les echecs, blocages, metriques et evenements.
"""
from __future__ import annotations

import re
import time

import streamlit as st

from core.models import resolve_tier_to_model_alias
from core.command_journal import render_command_journal
from core.multi_agent.background import (
    STATUS_CANCELLED as JOB_STATUS_CANCELLED,
    STATUS_ERROR as JOB_STATUS_ERROR,
    BackgroundJob,
)
from core.multi_agent.ledger import (
    STATE_CANCELLED,
    STATE_DISCOVERY,
    STATE_EXEC,
    STATE_INTERVIEW,
    STATE_PLANIF,
    STATE_REVIEW,
    STATE_SYNTHESE,
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_DOING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_REVIEW,
    STATUS_TODO,
)
from core.multi_agent.orchestrator import new_chief
from core.multi_agent.planning import estimate_plan_cost
from core.multi_agent.run_store import (
    list_runs,
    load_run,
    save_run,
    delete_run,
    RunStoreError,
)
from core.multi_agent.plan_preview import render_plan_preview_editor

# Mots vides à ignorer lors de l'extraction des termes de découverte.
_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "en",
    "dans", "pour", "avec", "sur", "par", "au", "aux", "ce", "cet", "cette",
    "ces", "son", "sa", "ses", "que", "qui", "quoi", "dont", "est", "sont",
    "puis", "peux", "peut", "tu", "il", "elle", "je", "nous", "vous", "ils",
    "the", "and", "for", "with", "this", "that", "voici", "réponse", "reponse",
    "confirme", "confirmer", "problème", "probleme", "mode", "lancé", "lance",
}




def _render_runs_sidebar() -> None:
    """Affiche la sidebar avec la liste des chantiers sauvegardés."""
    with st.sidebar:
        st.markdown("### Chantiers sauvegardés")
        
        try:
            runs = list_runs()
        except Exception as error:
            st.error(f"Erreur de chargement : {error}")
            return
        
        if not runs:
            st.caption("Aucun chantier sauvegardé.")
            return
        
        current_run_id = st.session_state.get(_KEY_CURRENT_RUN_ID)
        # Changer de chantier pendant un appel modèle provoquait un rerun qui
        # interrompait le traitement en cours : on verrouille la navigation.
        navigation_locked = _is_ma_api_request_pending()
        if navigation_locked:
            st.caption("Navigation verrouillee pendant l'operation en cours.")
        
        for run in runs[:15]:  # Limiter l'affichage à 15 chantiers
            run_id = run["id"]
            title = run["title"]
            state = run.get("state", "")
            state_label = _STATE_LABEL.get(state, state)
            
            # Indicateur visuel du chantier courant
            prefix = "▶ " if run_id == current_run_id else ""
            label = f"{prefix}{title[:40]}"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    label,
                    key=f"load_run_{run_id}",
                    help=f"État : {state_label}",
                    use_container_width=True,
                    disabled=navigation_locked,
                ):
                    try:
                        ledger = load_run(run_id)
                        if ledger is not None:
                            _restore_chief_from_ledger(ledger)
                            st.rerun()
                        else:
                            st.error(f"Chantier {run_id} introuvable.")
                    except RunStoreError as error:
                        st.error(f"Erreur de chargement : {error}")
            
            with col2:
                if st.button("🗑️", key=f"delete_run_{run_id}", help="Supprimer", disabled=navigation_locked):
                    if delete_run(run_id):
                        if run_id == current_run_id:
                            _reset_chief()
                        st.rerun()
                    else:
                        st.error("Suppression échouée.")
        
        if len(runs) > 15:
            st.caption(f"... et {len(runs) - 15} autre(s) chantier(s).")
        
        # Bouton de sauvegarde manuelle
        st.markdown("---")
        if st.button("💾 Sauvegarder maintenant", key="ma_manual_save", use_container_width=True):
            chief = st.session_state.get(_KEY_CHIEF)
            if chief:
                try:
                    chief.ledger.chief_history = list(chief.history)
                    save_run(chief.ledger)
                    st.success("Chantier sauvegardé.")
                except RunStoreError as error:
                    st.error(f"Sauvegarde échouée : {error}")
        
        # Toggle sauvegarde automatique
        auto_save = st.checkbox(
            "Sauvegarde automatique",
            value=st.session_state.get(_KEY_AUTO_SAVE_ENABLED, True),
            key="ma_auto_save_toggle",
        )
        st.session_state[_KEY_AUTO_SAVE_ENABLED] = auto_save
def _extract_discovery_terms(history) -> list[str]:
    """Extrait chemins, symboles techniques et mots discriminants."""
    message = next(
        (content or "" for role, content in reversed(history or []) if role == "user"),
        "",
    )
    terms: list[str] = []
    patterns = [
        r"(?:[\w.-]+/)+[\w.-]+",
        r"\b[\w.-]+\.(?:py|md|json|yaml|yml|toml)\b",
        r"\b[A-Z][A-Z0-9_]{2,}\b",
        r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, message):
            key = match.casefold()
            if key not in _STOPWORDS and all(key != item.casefold() for item in terms):
                terms.append(match)
            if len(terms) >= 12:
                return terms
    return terms

_KEY_CHIEF = "ma_chief"
_KEY_PLAN_REPORT = "ma_plan_report"
_KEY_LAST_REPLY = "ma_last_reply"
_KEY_OPERATION_STATUS = "ma_operation_status"
_KEY_OPERATION_TYPE = "ma_operation_type"
_KEY_OPERATION_ERROR = "ma_operation_error"

_KEY_CURRENT_RUN_ID = "ma_current_run_id"
_KEY_AUTO_SAVE_ENABLED = "ma_auto_save_enabled"
# Clés pour la machine à états API multi-agent (alignement sur app.py:926-985)
_KEY_API_STATUS = "ma_api_request_status"
_KEY_API_OPERATION = "ma_api_request_operation"
_KEY_API_ERROR = "ma_api_request_error"
# Job d'arrière-plan courant et compte rendu du dernier job terminé.
_KEY_JOB = "ma_background_job"
_KEY_JOB_REPORT = "ma_background_job_report"

# Cadence de rafraîchissement du moniteur, en secondes. Assez court pour que la
# progression paraisse vivante, assez long pour ne pas saturer le navigateur.
_MONITOR_REFRESH_SECONDS = 1.0
_MONITOR_VISIBLE_EVENTS = 25

_EVENT_ICON = {
    "state": "🔹",
    "plan": "🗺️",
    "worker": "⚙️",
    "error": "❌",
    "info": "ℹ️",
}


_STATUS_LABEL = {
    STATUS_TODO: "En attente",
    STATUS_DOING: "En cours",
    STATUS_DONE: "Terminee",
    STATUS_FAILED: "Echec",
    STATUS_BLOCKED: "Bloquee",
    STATUS_REVIEW: "A revoir",
    STATUS_CANCELLED: "Annulee",
}
_STATUS_ICON = {
    STATUS_TODO: "⏳",
    STATUS_DOING: "🔄",
    STATUS_DONE: "✅",
    STATUS_FAILED: "❌",
    STATUS_BLOCKED: "⛔",
    STATUS_REVIEW: "🔎",
    STATUS_CANCELLED: "🚫",
}
# Libelles lisibles des etats de chantier, utilises pour afficher l'etat global
# de facon constante (voir _render_run_summary).
_STATE_LABEL = {
    STATE_INTERVIEW: "Interview",
    STATE_DISCOVERY: "Decouverte",
    STATE_PLANIF: "Planification",
    STATE_EXEC: "Execution",
    STATE_REVIEW: "Revue",
    STATE_SYNTHESE: "Synthese",
    STATE_CANCELLED: "Annule",
}





# -------------------------------------------------------------------
# Fonctions de gestion d'état API multi-agent
# -------------------------------------------------------------------

def _begin_ma_api_request(operation: str) -> None:
    """Démarre une requête API multi-agent et persiste l'état d'attente."""
    st.session_state[_KEY_API_STATUS] = "pending"
    st.session_state[_KEY_API_OPERATION] = operation
    st.session_state[_KEY_API_ERROR] = None


def _complete_ma_api_request() -> None:
    """Termine une requête API multi-agent avec succès."""
    st.session_state[_KEY_API_STATUS] = "success"
    st.session_state[_KEY_API_OPERATION] = None
    st.session_state[_KEY_API_ERROR] = None


def _fail_ma_api_request(error: str) -> None:
    """Termine une requête API multi-agent avec une erreur."""
    st.session_state[_KEY_API_STATUS] = "error"
    st.session_state[_KEY_API_ERROR] = error
    st.session_state[_KEY_API_OPERATION] = None


def _reset_ma_api_request() -> None:
    """Réinitialise l'état de la requête API multi-agent."""
    st.session_state[_KEY_API_STATUS] = "idle"
    st.session_state[_KEY_API_OPERATION] = None
    st.session_state[_KEY_API_ERROR] = None


def _is_ma_api_request_pending() -> bool:
    """Vérifie si une requête API multi-agent est en cours."""
    job = st.session_state.get(_KEY_JOB)
    if job is not None and job.is_running():
        return True
    return st.session_state.get(_KEY_API_STATUS, "idle") == "pending"


# -------------------------------------------------------------------
# Exécution en arrière-plan
# -------------------------------------------------------------------
#
# Streamlit réexécute le script à chaque interaction. Exécuter un appel Bedrock
# dans le flux du script le rend donc interruptible par n'importe quel clic, et
# muet jusqu'à son terme. Les opérations longues sont déportées dans un thread
# (core/multi_agent/background.py) et le script se contente d'observer.


def _current_job() -> BackgroundJob | None:
    return st.session_state.get(_KEY_JOB)


def _make_threadsafe_save(chief, enabled: bool):
    """Construit un callback de sauvegarde utilisable depuis un thread.

    `_save_current_run` lit et écrit `st.session_state`, qui n'existe pas dans
    un thread de travail. On fige donc l'état du toggle au démarrage du job et
    on n'appelle plus que `save_run`, qui est du pur I/O disque.
    """

    def _save() -> None:
        if not enabled:
            return
        try:
            chief.ledger.chief_history = list(chief.history)
            save_run(chief.ledger)
        except Exception:
            # Une sauvegarde ratée ne doit jamais interrompre le chantier ;
            # l'incident reste visible dans les événements du job.
            pass

    return _save


def _start_background_job(chief, operation: str, target) -> None:
    """Démarre `target(job)` en arrière-plan et bascule l'UI en observation."""
    if _is_ma_api_request_pending():
        st.warning("Une operation est deja en cours.")
        return
    job = BackgroundJob(operation)
    chief.set_progress_callback(job.emit)
    chief.set_save_callback(
        _make_threadsafe_save(chief, bool(st.session_state.get(_KEY_AUTO_SAVE_ENABLED, True)))
    )
    st.session_state[_KEY_JOB] = job
    st.session_state[_KEY_JOB_REPORT] = None
    _begin_ma_api_request(operation)
    job.start(target)
    st.rerun()


def _finalize_job(chief) -> bool:
    """Applique le résultat d'un job terminé. Retourne True si un rerun s'impose.

    Tout ce qui touche `st.session_state` est fait ici, dans le thread du
    script, jamais dans le thread de travail.
    """
    job = _current_job()
    if job is None or not job.is_finished():
        return False

    events = job.drain()
    chief.set_progress_callback(None)
    chief.set_save_callback(None)
    st.session_state[_KEY_JOB] = None
    st.session_state[_KEY_JOB_REPORT] = {
        "operation": job.operation,
        "status": job.status,
        "error": job.error,
        "traceback": job.traceback,
        "elapsed": job.elapsed_seconds(),
        "events": events,
    }

    if job.status == JOB_STATUS_ERROR:
        _fail_ma_api_request(job.error or "Erreur inconnue.")
    else:
        _complete_ma_api_request()
        result = job.result if isinstance(job.result, dict) else {}
        if "plan_report" in result:
            st.session_state[_KEY_PLAN_REPORT] = result["plan_report"]
        if "final_answer" in result:
            st.session_state[_KEY_LAST_REPLY] = result["final_answer"]
        if result.get("state") is not None:
            chief.ledger.state = result["state"]

    _save_current_run(chief)
    return True


def _render_job_report() -> None:
    """Affiche le compte rendu du dernier job terminé."""
    report = st.session_state.get(_KEY_JOB_REPORT)
    if not report:
        return
    operation = report.get("operation", "operation")
    elapsed = report.get("elapsed", 0.0)
    status = report.get("status")

    if status == JOB_STATUS_ERROR:
        st.error(
            f"Echec de l'operation « {operation} » apres {elapsed:.0f}s : "
            f"{report.get('error')}"
        )
        if report.get("traceback"):
            with st.expander("Details techniques", expanded=False):
                st.code(report["traceback"])
    elif status == JOB_STATUS_CANCELLED:
        st.warning(
            f"Operation « {operation} » interrompue apres {elapsed:.0f}s. "
            "Les livrables deja produits sont conserves."
        )
    else:
        st.success(f"Operation « {operation} » terminee en {elapsed:.0f}s.")

    events = report.get("events") or []
    if events:
        with st.expander(f"Journal de l'operation ({len(events)} evenements)", expanded=False):
            for event in events:
                st.caption(f"`{event.kind}` {event.message}")


def _job_monitor_body() -> None:
    """Contenu rafraîchi en continu pendant l'exécution d'un job."""
    job = _current_job()
    chief = st.session_state.get(_KEY_CHIEF)
    if job is None or chief is None:
        return

    events = job.drain()
    elapsed = job.elapsed_seconds()

    header = f"Operation « {job.operation} » en cours — {elapsed:.0f}s"
    if job.cancel_event.is_set() and job.is_running():
        st.warning(f"{header} — interruption demandee, arret apres l'etape en cours.")
    elif job.is_running():
        st.info(f"{header}")
    elif job.status == JOB_STATUS_ERROR:
        st.error("Exécution interrompue par une erreur ; le travail enregistré est conservé.")
    elif job.status == JOB_STATUS_CANCELLED:
        st.warning("Exécution suspendue ; le travail enregistré est conservé.")
    else:
        st.success(f"Opération terminée en {elapsed:.0f}s.")

    tasks = chief.ledger.tasks
    if tasks:
        completed = sum(task.status == STATUS_DONE for task in tasks)
        st.progress(completed / len(tasks), text=f"{completed} / {len(tasks)} tâches terminées")
        _render_live_tasks(chief)
    elif job.is_running():
        st.caption("Analyse en cours…")
    if events:
        st.caption(events[-1].message)
    with st.expander("Détails de l’exécution", expanded=False):
        for event in reversed(events[-_MONITOR_VISIBLE_EVENTS:]):
            st.caption(event.message)

    if job.is_running() and not job.cancel_event.is_set():
        if st.button("Interrompre l'operation", key="ma_cancel_job"):
            job.request_cancel()


def _pulse_value(elapsed: float) -> float:
    """Barre d'activité : la durée réelle d'un appel Bedrock est inconnue.

    On n'affiche donc pas une fausse progression linéaire vers 100 %, mais une
    oscillation qui indique seulement que le traitement est vivant.
    """
    cycle = elapsed % 20.0
    return (cycle / 20.0) if cycle < 10.0 else ((20.0 - cycle) / 20.0)


def _render_live_tasks(chief) -> None:
    """Tableau compact des tâches, rafraîchi pendant l'exécution."""
    tasks = getattr(chief.ledger, "tasks", [])
    if not tasks:
        return
    rows = [
        {
            "Tâche": task.title,
            "Etat": f"{_STATUS_ICON.get(task.status, '❔')} {_STATUS_LABEL.get(task.status, task.status)}",

        }
        for task in tasks
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


if hasattr(st, "fragment"):

    @st.fragment(run_every=_MONITOR_REFRESH_SECONDS)
    def _render_job_monitor() -> None:
        _job_monitor_body()
        job = _current_job()
        if job is not None and job.is_finished():
            # Le job est terminé : on sort du fragment pour que le script
            # principal applique le résultat via _finalize_job.
            st.rerun(scope="app")

else:  # Streamlit < 1.37 : repli sur un rafraîchissement page entière.

    def _render_job_monitor() -> None:
        _job_monitor_body()
        job = _current_job()
        if job is None or not job.is_finished():
            time.sleep(_MONITOR_REFRESH_SECONDS)
        st.rerun()


def _reset_chief() -> None:
    """Crée un nouveau chantier vierge."""
    chief = new_chief()
    st.session_state[_KEY_CHIEF] = chief
    st.session_state[_KEY_PLAN_REPORT] = None
    st.session_state[_KEY_LAST_REPLY] = None
    st.session_state[_KEY_CURRENT_RUN_ID] = chief.ledger.run_id
    st.session_state[_KEY_AUTO_SAVE_ENABLED] = True
    # Initialisation de la machine à états API
    _reset_ma_api_request()



def _get_chief():
    """Retourne le chef courant, en le restaurant depuis le store si nécessaire."""
    if st.session_state.get(_KEY_CHIEF) is None:
        # Tentative de restauration depuis le run_id courant
        run_id = st.session_state.get(_KEY_CURRENT_RUN_ID)
        if run_id:
            try:
                ledger = load_run(run_id)
                if ledger is not None:
                    _restore_chief_from_ledger(ledger)
                    return st.session_state[_KEY_CHIEF]
            except RunStoreError as error:
                st.error(f"Impossible de restaurer le chantier : {error}")
        else:
            # Restauration automatique du dernier chantier au démarrage
            # (alignement sur le comportement du chat normal)
            try:
                runs = list_runs()
                if runs:
                    latest_run_id = runs[0]["id"]
                    ledger = load_run(latest_run_id)
                    if ledger is not None:
                        _restore_chief_from_ledger(ledger)
                        return st.session_state[_KEY_CHIEF]
            except RunStoreError:
                pass
        # Sinon, créer un nouveau chantier
        _reset_chief()
    return st.session_state[_KEY_CHIEF]


def _restore_chief_from_ledger(ledger: "TaskLedger") -> None:
    """Restaure un ChefDeProjet depuis un TaskLedger chargé."""
    chief = new_chief()
    ledger.recover_interrupted_tasks()
    chief.ledger = ledger
    # Restaurer chief_history depuis le ledger
    chief.history = list(ledger.chief_history) if hasattr(ledger, "chief_history") else []
    # Restaurer discovery_report si les données sont présentes
    if hasattr(ledger, "discovery_context") and ledger.discovery_context:
        from core.multi_agent.orchestrator import DiscoveryReport
        report = DiscoveryReport(restored=True)
        report.context_summary = ledger.discovery_context
        if hasattr(ledger, "discovered_files"):
            report.explored_files = list(ledger.discovered_files)
        if hasattr(ledger, "discovery_errors"):
            report.errors = list(ledger.discovery_errors)
        chief.discovery_report = report
    st.session_state[_KEY_CHIEF] = chief
    st.session_state[_KEY_CURRENT_RUN_ID] = ledger.run_id
    st.session_state[_KEY_AUTO_SAVE_ENABLED] = True


def _save_current_run(chief: "ChefDeProjet", title: str | None = None) -> None:
    """Sauvegarde le chantier courant si la sauvegarde automatique est activée."""
    if not st.session_state.get(_KEY_AUTO_SAVE_ENABLED, True):
        return
    try:
        # Synchroniser chief_history dans le ledger avant sauvegarde
        chief.ledger.chief_history = list(chief.history)
        save_run(chief.ledger, title=title)
        st.session_state[_KEY_CURRENT_RUN_ID] = chief.ledger.run_id
    except RunStoreError as error:
        st.warning(f"Sauvegarde automatique échouée : {error}")


def render_multi_agent() -> None:
    st.title("Mode multi-agent")
    st.caption(
        "Planification supervisee, execution parallele, revue humaine et "
        "synthese transparente."
    )
    
    # Sidebar : navigation entre chantiers
    _render_runs_sidebar()
    
    if st.button("Nouveau chantier", key="ma_new_run", disabled=_is_ma_api_request_pending()):
        # Sauvegarde du chantier courant avant d'en créer un nouveau
        # (alignement sur le comportement du chat normal, app.py:1161-1163)
        current_chief = st.session_state.get(_KEY_CHIEF)
        if current_chief:
            _save_current_run(current_chief)
        _reset_chief()
        st.rerun()

    chief = _get_chief()

    # Un job terminé est finalisé AVANT tout rendu de phase : l'état du chantier
    # doit être à jour quand l'interface se redessine.
    if _finalize_job(chief):
        st.rerun()

    # Pendant une opération, l'interface se limite au moniteur : aucun widget de
    # phase n'est exposé, donc aucun clic ne peut plus interrompre le travail.
    if _is_ma_api_request_pending():
        _render_run_summary(chief)
        _render_job_monitor()
        return

    _render_job_report()

    # Sauvegarde automatique après restauration (garantit la persistance
    # du run_id et des métadonnées même si aucune action n'est effectuée)
    _save_current_run(chief)
    
    _render_run_summary(chief)
    state = chief.ledger.state

    if state == STATE_INTERVIEW:
        _phase_interview(chief)
    elif state == STATE_DISCOVERY:
        _phase_discovery(chief)
    elif state == STATE_PLANIF:
        _phase_planif(chief)
    elif state == STATE_EXEC:
        # planning.build_ledger produit actuellement STATE_EXEC. Tant qu'aucune
        # tache n'a demarre, il s'agit encore de la revue humaine du plan.
        if all(task.status == STATUS_TODO for task in chief.ledger.tasks):
            _phase_planif(chief)
        else:
            _phase_review(chief)
    elif state == STATE_REVIEW:
        _phase_review(chief)
    elif state == STATE_SYNTHESE:
        _phase_synthesis(chief)
    elif state == STATE_CANCELLED:
        st.warning("Ce chantier a ete annule.")
        _render_task_board(chief)
    else:
        st.error(f"Etat multi-agent inconnu : {state!r}")

    render_command_journal(key_prefix="ma")
    _render_events(chief)


def _phase_discovery(chief) -> None:
    st.subheader("Découverte du workspace")
    st.info(
        "Le système explore automatiquement le workspace pour construire un "
        "contexte compact avant la planification."
    )
    
    if chief.discovery_report:
        report = chief.discovery_report
        if report.successful:
            st.success(
                f"Découverte valide : {len(report.file_contents)} preuve(s) de lecture, "
                f"{report.tool_calls} appel(s)."
            )
        else:
            st.error("Découverte invalide : aucun contenu de fichier vérifiable.")
        
        with st.expander("Détails de la découverte", expanded=False):
            if report.workspace_info:
                st.markdown("**Informations workspace**")
                st.code(report.workspace_info, language="text")
            
            if report.explored_files:
                st.markdown(f"**Fichiers identifiés** ({len(report.explored_files)})")
                for f in report.explored_files[:10]:
                    st.caption(f"- {f}")
            
            if report.search_results:
                st.markdown("**Recherches effectuées**")
                for query, result in list(report.search_results.items())[:3]:
                    with st.expander(f"Recherche : {query}"):
                        st.text(result[:500])
            
            if report.file_contents:
                st.markdown(f"**Preuves de lecture** ({len(report.file_contents)})")
                for path in list(report.file_contents.keys())[:5]:
                    st.caption(f"- {path}")
        
        if report.errors:
            with st.expander("Erreurs de découverte", expanded=not report.successful):
                for error in report.errors:
                    st.error(error)
        if st.button("Relancer la découverte", key="ma_retry_discovery"):
            terms = _extract_discovery_terms(chief.history)

            def _run_discovery(job):
                chief.discovery_report = None
                chief.discover(terms=terms if terms else None)
                return {}

            _start_background_job(chief, "decouverte", _run_discovery)
        if st.button(
            "Continuer vers la planification",
            key="ma_discovery_to_plan",
            type="primary",
            disabled=not report.successful,
        ):
            def _run_plan(job):
                job.emit("state", "Construction du plan...")
                plan_report = chief.plan(discovery_context=report.to_context())
                return {"plan_report": plan_report, "state": STATE_PLANIF}

            _start_background_job(chief, "planification", _run_plan)
    else:
        st.warning("Aucune découverte n'a encore été effectuée.")
        
        # Extraction de termes depuis l'historique (robuste, sans liste blanche figée).
        terms = _extract_discovery_terms(chief.history)
        
        if st.button(
            "Lancer la découverte automatique",
            key="ma_start_discovery",
            type="primary",
        ):
            def _run_first_discovery(job):
                chief.discover(terms=terms if terms else None)
                return {}

            _start_background_job(chief, "decouverte", _run_first_discovery)


def _phase_interview(chief) -> None:
    st.info("Decris ton besoin. Le Chef clarifie les points qui changent le plan.")
    for role, content in chief.history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    message = st.chat_input("Decris ou precise ton besoin...", key="ma_interview_input")
    if message:
        text = message
        def _run_interview(job):
            # La demande initiale sert directement à la recherche ciblée :
            # aucune reformulation LLM supplémentaire n'est nécessaire.
            prompt = text
            if not chief.discovery_report:
                terms = _extract_discovery_terms([("user", prompt)])
                job.emit("state", "Lecture ciblée des consignes et fichiers du projet...")
                try:
                    chief.discover(terms=terms or None)
                except Exception as error:
                    # L'entretien reste possible si le workspace est vide ou
                    # si la lecture de contexte échoue.
                    job.emit(
                        "info",
                        f"Découverte ignorée pour l'entretien : {type(error).__name__}.",
                    )
            job.emit("state", "Le Chef analyse la demande et le contexte du projet...")
            report = chief.prepare_request(prompt)
            return {"plan_report": report, "state": STATE_PLANIF} if report else {}

        _start_background_job(chief, "interview", _run_interview)

    if chief.history and st.button(
        "Analyser le workspace et préparer le plan",
        key="ma_discover_and_plan",
        type="primary",
    ):
        terms = _extract_discovery_terms(chief.history)

        def _run_prepare(job):
            job.emit(
                "state",
                f"Termes de decouverte retenus : {terms or 'fallback automatique'}.",
            )
            report = (
                chief.discovery_report
                if chief.discovery_report and chief.discovery_report.successful
                else chief.discover(terms=terms if terms else None)
            )
            job.emit(
                "state",
                f"{len(report.explored_files)} fichier(s) identifie(s), "
                f"{len(report.file_contents)} reellement lu(s).",
            )
            if not report.successful:
                raise RuntimeError(
                    "La decouverte n'a lu aucun fichier. La planification est "
                    "interrompue afin d'eviter un plan fonde sur un contexte vide."
                )
            if job.cancel_event.is_set():
                return {}
            job.emit("state", "Construction du plan...")
            plan_report = chief.plan(discovery_context=report.to_context())
            return {"plan_report": plan_report, "state": STATE_PLANIF}

        _start_background_job(chief, "preparation du chantier", _run_prepare)


def _phase_planif(chief) -> None:
    report = st.session_state.get(_KEY_PLAN_REPORT)
    st.subheader("Plan propose")
    if report is not None:
        st.caption(f"{report.task_count} tache(s), repartition : {report.tiers_used}")
        with st.expander("Estimation et répartition des modèles", expanded=False):
            st.caption(f"Indice relatif : {estimate_plan_cost(chief.ledger)}. Ce n’est ni un prix ni un plafond de facturation.")
            _render_team_structure(chief)
        assumptions = getattr(report, "assumptions", [])
        if assumptions:
            with st.expander("Hypotheses de planification", expanded=True):
                for assumption in assumptions:
                    st.markdown(f"- {assumption}")
        for correction in report.corrections:
            st.warning(f"Ajustement automatique : {correction}")
        for warning in getattr(report, "validation_warnings", []):
            st.info(warning)

    _render_task_board(chief)
    
    # Composant de prévisualisation et d'édition du plan
    # Affiche les prompts réels (système + tâche) pour chaque worker et permet
    # l'édition des attributs (title, description, tier, criteria, etc.).
    # Les prompts affichés sont calculés via les mêmes fonctions que l'orchestrateur
    # (worker_system_prompt, worker_task_prompt), garantissant la cohérence.
    with st.expander("🔍 Prévisualiser et éditer le plan", expanded=False):
        discovery_ctx = ""
        if chief.discovery_report:
            discovery_ctx = chief.discovery_report.to_context()
        
        edited_ledger = render_plan_preview_editor(chief.ledger, discovery_ctx)
        
        # Si l'utilisateur a validé des modifications, on met à jour le ledger
        # du chef et on rafraîchit l'interface pour refléter les changements.
        if edited_ledger is not None:
            chief.ledger = edited_ledger
            _save_current_run(chief)
            st.success("Plan mis à jour avec succès.")
            st.rerun()
    
    left, right = st.columns(2)
    with left:
        if st.button(
            "Valider le plan et lancer l'execution",
            key="ma_execute_plan",
            type="primary",
            use_container_width=True,
        ):
            def _run_execution(job):
                chief.execute(cancel_event=job.cancel_event)
                return {"state": STATE_REVIEW}

            _start_background_job(chief, "execution", _run_execution)
    with right:
        if st.button(
            "Revenir a l'interview",
            key="ma_back_to_interview",
            use_container_width=True,
        ):
            chief.ledger.state = STATE_INTERVIEW
            # Sauvegarde automatique après changement d'état
            _save_current_run(chief)
            st.rerun()


def _phase_review(chief) -> None:
    st.subheader("Revue des resultats")
    st.info(
        "Controle les livrables, les echecs et les blocages avant de demander "
        "la synthese finale."
    )
    _render_task_board(chief)

    chief.ledger.sync_artifact_states()
    resumed = chief.ledger.ready_tasks()
    stalled = chief.ledger.stalled_tasks()
    barrier_blocked = chief.ledger.barrier_blocked_tasks()
    if resumed:
        if st.button(
            f"Reprendre les tâches débloquées ({len(resumed)})",
            key="ma_resume_ready",
            type="primary",
        ):
            def _run_resume(job):
                chief.execute(cancel_event=job.cancel_event)
                return {"state": STATE_REVIEW}

            _start_background_job(chief, "reprise après synchronisation", _run_resume)

    if stalled:
        with st.expander(
            f"{len(stalled)} tâche(s) en attente : pourquoi ?", expanded=True
        ):
            for task, reason in stalled:
                st.markdown(f"**{task.title}** — {reason}")
        if barrier_blocked:
            st.info(
                "Ces tâches attendent une modification ou un contrôle validé. "
                "Consulte le motif ci-dessus. Lever cette condition permet "
                "de poursuivre sans cette garantie."
            )
            if st.button(
                f"Lever les barrières et reprendre ({len(barrier_blocked)})",
                key="ma_release_barriers",
            ):
                released = chief.ledger.release_barriers()

                def _run_released(job):
                    job.emit(
                        "state",
                        f"{len(released)} barriere(s) levee(s) : reprise de l'execution.",
                    )
                    chief.execute(cancel_event=job.cancel_event)
                    return {"state": STATE_REVIEW}

                _start_background_job(chief, "reprise apres levee de barriere", _run_released)

    retryable = [
        task for task in chief.ledger.tasks
        if task.status in {STATUS_FAILED, STATUS_BLOCKED, STATUS_REVIEW}
    ]
    if retryable:
        st.warning(f"{len(retryable)} tache(s) necessitent une action.")
        if st.button("Relancer les taches eligibles", key="ma_retry_all"):
            reset_count = 0
            for task in retryable:
                try:
                    chief.ledger.retry_task(task.id)
                    reset_count += 1
                except ValueError:
                    continue
            if not reset_count:
                st.warning(
                    "Aucune tache n'a pu etre relancee : leurs dependances ne "
                    "sont pas satisfaites."
                )
                _save_current_run(chief)
                st.rerun()

            def _run_retry(job):
                chief.execute(cancel_event=job.cancel_event)
                return {"state": STATE_REVIEW}

            _start_background_job(chief, "relance ciblee", _run_retry)

    # Revue croisee : les taches marquees requires_review par le plan sont
    # critiquees par un agent d'un tier superieur avant la synthese.
    pending_review = [
        task for task in chief.ledger.tasks
        if getattr(task, "requires_review", False)
        and task.deliverable
        and not getattr(task, "review_verdict", None)
    ]
    if pending_review:
        st.caption(
            f"{len(pending_review)} livrable(s) sont marques pour revue croisee : "
            "un agent critique d'un tier superieur verifie la conformite aux "
            "criteres d'acceptation."
        )
        if st.button(
            f"Lancer la revue croisee ({len(pending_review)})",
            key="ma_cross_review",
        ):
            def _run_review(job):
                verdicts = chief.review_flagged_tasks(cancel_event=job.cancel_event)
                refused = [item for item in verdicts if item[1] == "a corriger"]
                job.emit(
                    "state",
                    f"Revue terminee : {len(refused)} livrable(s) juges non conformes "
                    f"sur {len(verdicts)}.",
                )
                return {}

            _start_background_job(chief, "revue croisee", _run_review)

    reviewed = [
        task for task in chief.ledger.tasks
        if getattr(task, "review_comment", "")
    ]
    if reviewed:
        with st.expander(f"Critiques des agents ({len(reviewed)})"):
            for task in reviewed:
                st.markdown(f"**{task.title}** — verdict : `{task.review_verdict}`")
                st.text(task.review_comment)

    # La synthèse est ouverte dès qu'aucune tâche n'est plus exécutable
    # automatiquement. Une tâche TODO qui n'est ni prête ni en cours est
    # bloquée par une barrière ou une dépendance en échec : elle ne progressera
    # pas sans action humaine et ne doit donc plus condamner le chantier.
    stalled_ids = {task.id for task, _ in stalled}
    actionable = [
        task for task in chief.ledger.tasks
        if task.status == STATUS_DOING
        or (task.status == STATUS_TODO and task.id not in stalled_ids)
    ]
    if chief.ledger.tasks and not actionable:
        incidents = [
            task for task in chief.ledger.tasks
            if task.status in {STATUS_FAILED, STATUS_BLOCKED, STATUS_CANCELLED, STATUS_REVIEW}
            or task.id in stalled_ids
        ]
        if incidents:
            st.info(
                f"{len(incidents)} tache(s) n'ont pas abouti. La synthese reste "
                "possible : elle sera explicitement presentee comme partielle."
            )
        label = (
            "Produire la synthese finale"
            if not incidents
            else "Produire la synthese partielle"
        )
        if st.button(label, key="ma_synthesize", type="primary"):
            def _run_synthesis(job):
                job.emit("state", "Synthese et controle des contradictions...")
                return {"final_answer": chief.synthesize()}

            _start_background_job(chief, "synthese", _run_synthesis)

    if st.button("Annuler le chantier", key="ma_cancel_run"):
        chief.ledger.request_cancellation("Annulation demandee depuis l'interface.")
        # Sauvegarde automatique après annulation
        _save_current_run(chief)
        st.rerun()


def _phase_synthesis(chief) -> None:
    st.subheader("Reponse finale")
    reply = st.session_state.get(_KEY_LAST_REPLY) or chief.ledger.final_answer
    if reply:
        st.markdown(reply)
    else:
        st.warning("Aucune synthese disponible.")

    _render_status_recap(chief)
    with st.expander("Detail du travail de l'equipe", expanded=False):
        _render_team_structure(chief)
        _render_task_board(chief)

    feedback = st.chat_input(
        "Un retour ? Le Chef preparera un nouveau plan.",
        key="ma_feedback_input",
    )
    if feedback:
        def _run_replan(job):
            job.emit("state", "Prise en compte du retour utilisateur...")
            chief.apply_feedback(feedback)
            job.emit("state", "Nouveau plan en cours de construction...")
            report = chief.plan()
            return {"plan_report": report, "state": STATE_PLANIF, "final_answer": None}

        st.session_state[_KEY_LAST_REPLY] = None
        _start_background_job(chief, "re-planification", _run_replan)


def _render_run_summary(chief) -> None:
    ledger = chief.ledger
    summary = ledger.metrics_summary() if hasattr(ledger, "metrics_summary") else {}
    if summary.get("unknown_usage_calls"):
        st.caption(f"Usage incomplet pour {summary['unknown_usage_calls']} appel(s) interrompu(s) : "
                   "leur coût éventuel n'est pas compté comme nul.")
    if summary.get("coordination_model_calls"):
        st.caption(f"Les totaux incluent {summary['coordination_model_calls']} appel(s) d'entretien, "
                   "de planification, de revue ou de synthèse.")
    # Visibilite constante de l'etat global du chantier (INTERVIEW, EXEC,
    # CANCELLED, etc.) : affiche hors expander pour rester lisible dans toutes
    # les phases, y compris apres une annulation ou une synthese.
    state_label = _STATE_LABEL.get(ledger.state, ledger.state)
    st.caption(f"**Etat du chantier : {state_label}**")
    if getattr(ledger, "cancel_requested", False):
        reason = getattr(ledger, "cancellation_reason", None)
        st.warning(
            "Chantier annule" + (f" : {reason}" if reason else ".")
        )
    with st.expander("Informations du chantier", expanded=False):
        st.caption(f"Run ID : {getattr(ledger, 'run_id', 'indisponible')}")
        st.caption(f"Etat : {ledger.state}")
        # Un compteur par etat terminal/actif afin que "annulees" soit visible
        # au meme titre que "terminees", "echecs" et "bloquees".
        columns = st.columns(6)
        columns[0].metric("Taches", summary.get("task_count", len(ledger.tasks)))
        columns[1].metric("Terminees", summary.get("done", 0))
        columns[2].metric("Echecs", summary.get("failed", 0))
        columns[3].metric("Bloquees", summary.get("blocked", 0))
        columns[4].metric("Annulees", summary.get("cancelled", 0))
        columns[5].metric("Tokens cumulés", summary.get("total_tokens", 0),
                          help="Entrées et sorties de tous les appels connus. Un historique relu est compté à chaque appel.")
        usage = st.columns(3)
        usage[0].metric("Appels modèles", summary.get("model_calls", 0))
        usage[1].metric("Tokens en entrée", summary.get("input_tokens", 0))
        usage[2].metric("Tokens en sortie", summary.get("output_tokens", 0))
        ordinary = max(0, summary.get("input_tokens", 0) - summary.get("cache_read_tokens", 0)
                       - summary.get("cache_write_tokens", 0))
        st.caption(f"Entrées hors cache : {ordinary}. "
                   f"Cache : {summary.get('cache_read_tokens', 0)} tokens lus, "
                   f"{summary.get('cache_write_tokens', 0)} écrits. Les prix dépendent du modèle.")
        rows = []
        for task in ledger.tasks:
            metrics = task.metrics
            rows.append({"Étape": task.title, "Modèle": task.assigned_tier,
                         "Appels": metrics.model_calls, "Entrée": metrics.input_tokens or 0,
                         "Sortie": metrics.output_tokens or 0, "Cache lu": metrics.cache_read_tokens})
        metrics = ledger.coordination_metrics
        rows.append({"Étape": "Coordination", "Modèle": "chef / revue",
                     "Appels": metrics.model_calls, "Entrée": metrics.input_tokens or 0,
                     "Sortie": metrics.output_tokens or 0, "Cache lu": metrics.cache_read_tokens})
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption("Les fichiers lus sont des fichiers distincts ; ce nombre ne mesure pas "
                   "les relectures du contexte. Les artefacts contrôlés ne correspondent pas "
                   "à une couverture complète du code.")


def _render_team_structure(chief) -> None:
    st.markdown("#### Equipe mobilisee")
    grouped = chief.ledger.team_structure()
    if not grouped:
        st.caption("Aucune tache assignee.")
        return
    # Chaque tache correspond a un agent worker distinct (1 tache = 1 worker,
    # cf. workers.py). Le nombre d'agents est donc le nombre total de taches,
    # et non le nombre de tiers.
    total_agents = sum(len(tasks) for tasks in grouped.values())
    tier_resume = ", ".join(
        f"{tier} : {len(tasks)}" for tier, tasks in grouped.items()
    )
    st.markdown(
        f"**{total_agents} agent(s)** repartis sur {len(grouped)} tier(s) "
        f"({tier_resume})."
    )

    # Vue unique et lisible de l'effectif : un agent par ligne, avec son
    # modele reel, son etat, ses dependances et son cout. Les colonnes par
    # tier precedentes rendaient l'equipe illisible des 4 tiers.
    alias_cache: dict[str, str] = {}

    def _alias(tier: str) -> str:
        if tier not in alias_cache:
            try:
                alias_cache[tier] = resolve_tier_to_model_alias(tier)
            except Exception:
                alias_cache[tier] = "?"
        return alias_cache[tier]

    rows = []
    for tier, tasks in grouped.items():
        for task in tasks:
            metrics = getattr(task, "metrics", None)
            rows.append({
                "Agent": getattr(task, "id", "?"),
                "Etat": (
                    f"{_STATUS_ICON.get(task.status, '❔')} "
                    f"{_STATUS_LABEL.get(task.status, task.status)}"
                ),
                "Tache": getattr(task, "title", "?"),
                "Tier": tier,
                "Modele": _alias(tier),
                "Depend de": ", ".join(getattr(task, "dependencies", []) or []) or "—",
                "Barrière": getattr(task, "dependency_state", "completed"),
                "Revue": (
                    getattr(task, "review_verdict", None)
                    or ("attendue" if getattr(task, "requires_review", False) else "—")
                ),
                "Tokens": getattr(metrics, "total_tokens", None) or 0,
            })
    rows.sort(key=lambda row: str(row["Agent"]))
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_status_recap(chief) -> None:
    """Affiche un recapitulatif compact et constant des etats des taches.

    Reste lisible en un coup d'oeil (en cours / termine / annule et autres
    statuts existants), y compris lorsque les expandeurs individuels ou le
    tableau complet sont replies.
    """
    tasks = getattr(chief.ledger, "tasks", [])
    if not tasks:
        return
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.status] = counts.get(task.status, 0) + 1
    recap_parts = [
        f"{_STATUS_ICON.get(status, '❔')} {_STATUS_LABEL.get(status, status)} : {count}"
        for status, count in counts.items()
    ]
    if recap_parts:
        st.caption(" | ".join(recap_parts))


def _render_task_board(chief) -> None:
    if not chief.ledger.tasks:
        return
    st.markdown("#### Tableau des taches")
    # Recapitulatif constant des etats en tete du tableau : reste lisible meme
    # quand les expandeurs individuels des taches sont replies.
    _render_status_recap(chief)
    for task in chief.ledger.tasks:
        icon = _STATUS_ICON.get(task.status, "❔")
        label = _STATUS_LABEL.get(task.status, f"Etat inconnu : {task.status}")
        header = f"{icon} {task.title} — {label}"
        with st.expander(header, expanded=task.status in {STATUS_FAILED, STATUS_BLOCKED}):
            st.caption(task.description)
            if task.dependencies:
                st.caption("Dependances : " + ", ".join(task.dependencies))
            role = getattr(task, "assigned_role", None)
            if role:
                st.caption(f"Role : {role}")
            artifact = getattr(task, "artifact", None)
            if artifact:
                unapplied = any(value.startswith("Aucune écriture prouvée") for value in artifact.blockers)
                st.markdown(artifact.summary if unapplied else artifact.result or artifact.summary)
                if unapplied and artifact.result:
                    with st.expander("Contenu généré non appliqué", expanded=False):
                        st.code(artifact.result)
                labels = {"observed": "Résultat produit", "applied": "Modifications enregistrées · contrôles non validés",
                          "tested": "Contrôles réussis sur la version observée", "proposed": "Modifications proposées"}
                st.caption(labels.get(artifact.state, artifact.state))
                for blocker in artifact.blockers:
                    st.warning(blocker)
            elif task.deliverable:
                st.markdown(task.deliverable)
            error = getattr(task, "error", None)
            if error:
                st.error(error)
            notes = getattr(task, "notes", "")
            if notes:
                with st.expander("Notes techniques", expanded=False):
                    st.code(notes)
            with st.expander("Informations techniques", expanded=False):
                st.caption(f"Modèle : {task.assigned_tier} · Tentatives : {task.attempt_count}")
                st.caption(f"Appels : {task.metrics.model_calls} · Tokens connus : {task.metrics.total_tokens or 0}")
                if task.command:
                    st.code(" ".join(task.command))
                if artifact:
                    st.json({"references": artifact.references,
                             "evidence": [vars(item) for item in artifact.evidence]})


def _render_events(chief) -> None:
    events = list(getattr(chief, "events", []))
    ledger_events = list(getattr(chief.ledger, "events", []))
    if not events and not ledger_events:
        return
    with st.expander("Journal d'execution", expanded=False):
        for event in events:
            st.caption(f"[{event.kind}] {event.message}")
        for event in ledger_events:
            st.caption(f"[{event.level}/{event.kind}] {event.message}")


def _render_exception(message: str, error: Exception) -> None:
    st.error(message)
    with st.expander("Details techniques", expanded=False):
        st.exception(error)
