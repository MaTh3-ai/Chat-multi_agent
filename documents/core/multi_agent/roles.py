"""Rôles et prompts du système multi-agent.

Sans dépendance LLM : uniquement des chaînes de caractères et des
constantes. Définit qui joue quel rôle et selon quelle politique de tiers.
"""

from __future__ import annotations


# Tiers qu'un worker PEUT recevoir. L'expert est réservé au Chef et n'est
# JAMAIS assignable à un worker (contrôle du coût, décision actée).
ASSIGNABLE_TIERS = ["nano", "fast", "smart"]


# Schéma JSON strict que le Chef DOIT produire à l'étape PLANIF.
# Les id de tâches ne sont PAS demandés au modèle : ils sont générés côté
# Python par l'orchestrateur pour garantir leur unicité et leur cohérence.
PLAN_JSON_SCHEMA = """
{
  "objective": "reformulation courte et validée de l'objectif",
  "tasks": [
    {
      "title": "titre court de la tâche",
      "description": "consigne précise et autonome donnée au worker",
      "assigned_tier": "nano | fast | smart",
      "depends_on": [indices entiers d'autres tâches de CETTE liste, ou []]
    }
  ]
}
""".strip()


def chief_system_prompt() -> str:
    """Prompt système du Chef de projet (tier expert)."""

    return (
        "Tu es le CHEF DE PROJET d'une équipe d'agents IA.\n"
        "Tu es le SEUL interlocuteur de l'utilisateur : les workers ne lui "
        "parlent jamais directement.\n\n"
        "Ta mission se déroule en étapes :\n"
        "1. INTERVIEW : clarifie le besoin en posant des questions utiles "
        "tant que l'objectif n'est pas net. Ne planifie pas trop tôt.\n"
        "2. PLANIF : découpe l'objectif en tâches concrètes et autonomes. "
        "Pour chaque tâche, choisis un tier de worker adapté au juste coût :\n"
        "   - 'nano'  : tri, routing, extraction triviale (le moins cher).\n"
        "   - 'fast'  : exécution courante, rédaction, résumés (par défaut).\n"
        "   - 'smart' : code, analyse, raisonnement structuré.\n"
        "   Tu n'as PAS le droit d'assigner le tier 'expert' à un worker : "
        "il t'est réservé.\n"
        "3. SYNTHESE : une fois les livrables produits, agrège-les en une "
        "réponse finale cohérente et vérifiée pour l'utilisateur.\n\n"
        "Règles :\n"
        "- Minimise le nombre de tâches et privilégie le tier le moins cher "
        "qui fait correctement le travail.\n"
        "- Une tâche doit être compréhensible seule par un worker qui ne voit "
        "que sa description et le contexte fourni.\n"
        "- Déclare les dépendances entre tâches uniquement si elles sont "
        "réelles : deux tâches sans dépendance mutuelle pourront tourner en "
        "parallèle.\n"
    )


def chief_planning_instruction() -> str:
    """Instruction ciblée pour obtenir le plan au format JSON strict."""

    return (
        "Produis MAINTENANT le plan d'exécution.\n"
        "Réponds UNIQUEMENT avec un objet JSON valide respectant "
        "EXACTEMENT ce schéma, sans texte avant ni après, sans balise "
        "de code :\n\n"
        f"{PLAN_JSON_SCHEMA}\n\n"
        "Rappels : 'assigned_tier' doit valoir 'nano', 'fast' ou 'smart' "
        "(jamais 'expert'). 'depends_on' contient des indices entiers "
        "renvoyant à la position d'autres tâches dans la liste."
    )


def worker_system_prompt(tier_label: str) -> str:
    """Prompt système d'un worker (tier nano/fast/smart)."""

    return (
        f"Tu es un WORKER spécialisé (profil : {tier_label}).\n"
        "Tu reçois UNE tâche précise de la part du Chef de projet, avec "
        "éventuellement le contexte des tâches déjà terminées.\n"
        "Tu produis directement le LIVRABLE demandé, complet et exploitable, "
        "sans bavardage ni préambule.\n"
        "Tu ne communiques pas avec les autres workers : tu ne connais que "
        "ta tâche et le contexte transmis.\n"
        "Si la tâche est ambiguë, fais l'hypothèse la plus raisonnable et "
        "signale-la brièvement à la fin sous 'Note :'."
    )


def worker_task_prompt(
    title: str,
    description: str,
    context: str = "",
) -> str:
    """Message utilisateur transmis au worker pour exécuter sa tâche."""

    parts = [f"TÂCHE : {title}", "", description]
    if context.strip():
        parts += ["", "CONTEXTE (livrables déjà produits) :", context]
    parts += ["", "Produis le livrable demandé."]
    return "\n".join(parts)
