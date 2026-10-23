"""Validation et compilation déterministes d'un plan multi-agent."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.multi_agent.ledger import STATE_PLANIF, Task, TaskLedger
from core.models import get_tier
from core.multi_agent.roles import ASSIGNABLE_TIERS

MAX_TASKS = 24
MIN_TASKS = 1
MAX_TEXT_LENGTH = 30_000
MAX_FIELD_LENGTH = 8_000
VALID_CAPABILITIES = {"auto", "read", "write", "execute", "propose_write", "generate"}
# Classification included in the existing planning call; Python resolves the
# model. Legacy plans and explicit model choices keep assigned_tier priority.
WORK_KIND_TIERS = {"extraction": "nano", "routine": "fast", "complex": "smart", "critical": "expert"}
VALID_DEPENDENCY_STATES = {"completed", "applied", "tested", "verified"}
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Défauts JSON récurrents des sorties de modèles.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_LINE_COMMENT_RE = re.compile(r"(?<![:\"'\w])//[^\n]*")
_PYTHON_LITERAL_RE = re.compile(r"(?<![\"\w])(True|False|None)(?![\"\w])")
_PYTHON_LITERALS = {"True": "true", "False": "false", "None": "null"}


class PlanError(ValueError):
    """Plan invalide ou impossible à interpréter.

    ``semantic`` distingue une erreur de CONTENU (plan compréhensible mais non
    conforme aux règles : trop peu de tâches, dépendance impossible...) d'une
    erreur de FORMAT (JSON illisible). La réparation demandée au Chef diffère :
    reformater sans changer le sens, ou corriger réellement le plan.
    """

    def __init__(self, message: str, *, semantic: bool = False) -> None:
        super().__init__(message)
        self.semantic = semantic


@dataclass(frozen=True)
class PlanReport:
    task_count: int
    tiers_used: dict[str, int]
    corrections: list[str]
    assumptions: list[str]
    review_task_count: int = 0
    estimated_cost_units: int = 0
    validation_warnings: list[str] = field(default_factory=list)


def estimate_plan_cost(ledger: TaskLedger) -> int:
    """Estime des unités relatives au pire cas, sans prétendre à un coût AWS."""
    total = 0
    for task in ledger.tasks:
        tier_cost = get_tier(task.assigned_tier)["relative_cost"]
        retries_per_run = 1 + max(0, int(getattr(task, "max_retries", 1)))
        total += tier_cost * retries_per_run
        if task.requires_review:
            total += get_tier("fast" if task.assigned_tier == "nano" else task.assigned_tier)["relative_cost"]
    return total


def repair_common_json_defects(text: str) -> str:
    """Corrige les défauts JSON les plus fréquents des sorties de modèles.

    Traite, sans jamais changer le sens du plan :
    - les littéraux Python ``True`` / ``False`` / ``None`` ;
    - les virgules finales avant ``}`` ou ``]`` ;
    - les commentaires ``//`` en fin de ligne.

    La fonction reste volontairement conservatrice : elle n'essaie pas de
    deviner des guillemets manquants, ce qui produirait un plan faux.
    """
    repaired = _TRAILING_COMMA_RE.sub(r"\1", text)
    repaired = _LINE_COMMENT_RE.sub("", repaired)
    repaired = _PYTHON_LITERAL_RE.sub(
        lambda match: _PYTHON_LITERALS[match.group(0)], repaired
    )
    return repaired


def extract_json_block(raw: str) -> str:
    """Extrait le premier objet JSON décodable d'une réponse textuelle."""
    if not isinstance(raw, str) or not raw.strip():
        raise PlanError("Réponse de planification vide.")
    if len(raw) > MAX_TEXT_LENGTH:
        raise PlanError(
            f"Réponse de planification trop longue ({len(raw)} caractères)."
        )

    cleaned = raw.strip()
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    saw_opening_brace = False
    last_error: json.JSONDecodeError | None = None

    # Deuxième passe sur une version réparée : les modèles produisent
    # régulièrement des virgules finales ou des littéraux Python.
    for candidate in (cleaned, repair_common_json_defects(cleaned)):
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            saw_opening_brace = True
            try:
                _, end = decoder.raw_decode(candidate[index:])
                return candidate[index : index + end]
            except json.JSONDecodeError as exc:
                last_error = exc

    if not saw_opening_brace:
        raise PlanError("Aucun objet JSON détecté dans le plan.")

    preview = cleaned[:500].replace("\n", "\\n")
    truncation_hint = (
        " La réponse semble coupée avant la fin (accolade finale absente) : "
        "réduis le nombre de tâches ou raccourcis les descriptions."
        if cleaned.count("{") > cleaned.count("}")
        else ""
    )
    raise PlanError(
        "Objet JSON détecté mais invalide"
        + (f" : {last_error}" if last_error else "")
        + f".{truncation_hint} Aperçu : {preview!r}"
    )


def parse_plan(raw: str) -> dict[str, Any]:
    block = extract_json_block(raw)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as exc:
        preview = block[:500].replace("\n", "\\n")
        raise PlanError(f"JSON du plan invalide : {exc}. Aperçu : {preview!r}") from exc

    if not isinstance(data, dict):
        raise PlanError("Le plan doit être un objet JSON.")
    if set(data) - {"objective", "assumptions", "tasks"}:
        # Les champs supplémentaires de premier niveau sont tolérés mais ignorés.
        pass
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise PlanError(
            "Le plan doit contenir une liste 'tasks' non vide.", semantic=True
        )
    if len(tasks) < MIN_TASKS:
        raise PlanError(
            f"Le plan doit contenir au moins {MIN_TASKS} tâche. "
            f"Reçu : {len(tasks)}.",
            semantic=True,
        )
    if len(tasks) > MAX_TASKS:
        raise PlanError(
            f"Le plan dépasse la limite de {MAX_TASKS} tâches.", semantic=True
        )
    return data


def _make_task_id(index: int) -> str:
    return f"T{index + 1:02d}"


def _clean_text(value: object, field: str, where: str, required: bool = True) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise PlanError(f"{where} : '{field}' est obligatoire.", semantic=True)
    if len(text) > MAX_FIELD_LENGTH:
        raise PlanError(
            f"{where} : '{field}' dépasse {MAX_FIELD_LENGTH} caractères.", semantic=True
        )
    return text


def _normalize_tier(value: object, corrections: list[str], where: str) -> str:
    tier = str(value or "").strip().lower()
    if tier in ASSIGNABLE_TIERS:
        return tier
    corrections.append(f"{where} : tier {value!r} remplacé par 'fast'.")
    return "fast"


def _clean_string_list(value: object, field: str, where: str) -> list[str]:
    if value is None:
        return []
    # Une chaîne unique est acceptée et normalisée en liste d'un élément.
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise PlanError(f"{where} : '{field}' doit être une liste.", semantic=True)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item, field, where, required=False)
        key = text.casefold()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def _normalize_capabilities(
    value: object, corrections: list[str], where: str
) -> list[str]:
    capabilities = _clean_string_list(value, "capabilities", where)
    if not capabilities:
        return ["auto"]
    normalized = [item.casefold() for item in capabilities]
    invalid = [item for item in normalized if item not in VALID_CAPABILITIES]
    if invalid:
        corrections.append(
            f"{where} : capacités inconnues ignorées ({', '.join(invalid)})."
        )
    valid = [item for item in normalized if item in VALID_CAPABILITIES]
    if not valid:
        return ["auto"]
    if "auto" in valid and len(valid) > 1:
        valid.remove("auto")
    return list(dict.fromkeys(valid))


def _bounded_int(
    value: object,
    field_name: str,
    where: str,
    corrections: list[str],
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        value = None
    try:
        number = int(value) if value is not None else default
    except (TypeError, ValueError):
        number = default
    bounded = max(minimum, min(number, maximum))
    if value is not None and number != bounded:
        corrections.append(
            f"{where} : '{field_name}' borné à {bounded}."
        )
    return bounded


def _resolve_dependencies(
    value: object,
    self_index: int,
    task_count: int,
    where: str,
    corrections: list[str],
) -> list[int]:
    """Normalise depends_on en ignorant les entrées inexploitables.

    Une dépendance fantaisiste (indice hors limites, référence vers une tâche
    ultérieure, valeur non entière) ne justifie pas de rejeter tout le plan :
    elle est retirée et l'ajustement est tracé dans le rapport.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        corrections.append(f"{where} : 'depends_on' n'est pas une liste, ignoré.")
        return []
    valid: list[int] = []
    for dependency in value:
        if isinstance(dependency, bool) or not isinstance(dependency, int):
            corrections.append(
                f"{where} : dépendance {dependency!r} non entière, ignorée."
            )
            continue
        if dependency < 0 or dependency >= task_count:
            corrections.append(
                f"{where} : dépendance {dependency} hors limites, ignorée."
            )
            continue
        if dependency >= self_index:
            corrections.append(
                f"{where} : dépendance {dependency} vers une tâche ultérieure, ignorée."
            )
            continue
        if dependency not in valid:
            valid.append(dependency)
    return valid


def build_ledger(raw_plan: str) -> tuple[TaskLedger, PlanReport]:
    data = parse_plan(raw_plan)
    raw_tasks = data["tasks"]
    corrections: list[str] = []
    assumptions = _clean_string_list(data.get("assumptions"), "assumptions", "Plan")
    objective = _clean_text(data.get("objective"), "objective", "Plan")
    identifiers = {index: _make_task_id(index) for index in range(len(raw_tasks))}
    ledger = TaskLedger(objective=objective, state=STATE_PLANIF)
    tiers_used: dict[str, int] = {}
    review_count = 0

    for index, raw_task in enumerate(raw_tasks):
        where = f"Tâche {index + 1}"
        if not isinstance(raw_task, dict):
            raise PlanError(f"{where} : objet attendu.", semantic=True)

        title = _clean_text(raw_task.get("title"), "title", where)
        description = _clean_text(raw_task.get("description"), "description", where)
        expected_output = _clean_text(
            raw_task.get("expected_output"), "expected_output", where, required=False
        )
        if not expected_output:
            expected_output = "Livrable textuel répondant à l'instruction."
            corrections.append(f"{where} : 'expected_output' manquant, valeur par défaut appliquée.")
        criteria = _clean_string_list(
            raw_task.get("acceptance_criteria"), "acceptance_criteria", where
        )
        if not criteria:
            criteria = ["Le livrable répond entièrement à l'instruction de la tâche."]
            corrections.append(
                f"{where} : aucun critère d'acceptation fourni, critère générique appliqué."
            )

        work_kind = str(raw_task.get("work_kind", "")).strip().lower()
        requested_tier = raw_task.get("assigned_tier")
        if not requested_tier and work_kind in WORK_KIND_TIERS:
            requested_tier = WORK_KIND_TIERS[work_kind]
        tier = _normalize_tier(requested_tier, corrections, where)
        if work_kind:
            ledger.log_decision(f"{where} : travail {work_kind}, modèle {tier}.")
        capabilities = _normalize_capabilities(
            raw_task.get("capabilities"), corrections, where
        )
        max_retries = _bounded_int(
            raw_task.get("max_retries"),
            "max_retries",
            where,
            corrections,
            default=1,
            minimum=0,
            maximum=2,
        )
        max_escalations = _bounded_int(
            raw_task.get("max_escalations"),
            "max_escalations",
            where,
            corrections,
            default=1,
            minimum=0,
            maximum=1,
        )
        dependency_indices = _resolve_dependencies(
            raw_task.get("depends_on"), index, len(raw_tasks), where, corrections
        )
        dependency_state = str(
            raw_task.get("dependency_state", "completed")
        ).strip().casefold()
        if dependency_state not in VALID_DEPENDENCY_STATES:
            corrections.append(
                f"{where} : 'dependency_state' inconnu, remplacé par completed."
            )
            dependency_state = "completed"
        requires_review = raw_task.get("requires_review")
        if not isinstance(requires_review, bool):
            # Un champ absent ou mal typé ne doit pas invalider tout le plan :
            # l'absence de revue est le comportement par défaut le plus sûr.
            if requires_review is not None:
                corrections.append(
                    f"{where} : 'requires_review' non booléen, interprété comme false."
                )
            requires_review = False
        review_count += int(requires_review)

        command = raw_task.get("command", [])
        if not isinstance(command, list) or any(not isinstance(part, str) or not part.strip() for part in command):
            raise PlanError(f"{where}: command doit être une liste d'arguments.")
        if command and capabilities != ["execute"]:
            raise PlanError(f"{where}: command exige capabilities=[execute].", semantic=True)

        ledger.add_task(
            Task(
                id=identifiers[index],
                title=title,
                description=description,
                expected_output=expected_output,
                acceptance_criteria=criteria,
                assigned_tier=tier,
                capabilities=capabilities,
                command=command,
                max_retries=max_retries,
                max_escalations=max_escalations,
                requires_review=requires_review,
                dependencies=[identifiers[item] for item in dependency_indices],
                dependency_state=dependency_state,
            )
        )
        tiers_used[tier] = tiers_used.get(tier, 0) + 1

    _check_no_cycle(ledger)
    for assumption in assumptions:
        ledger.log_decision(f"Hypothèse de planification : {assumption}")

    estimated_cost_units = estimate_plan_cost(ledger)
    validation_warnings = [
        "L'estimation relative compte les reprises et les revues demandées ; "
        "le coût réel peut être inférieur.",
    ]
    return ledger, PlanReport(
        task_count=len(raw_tasks),
        tiers_used=tiers_used,
        corrections=corrections,
        assumptions=assumptions,
        review_task_count=review_count,
        estimated_cost_units=estimated_cost_units,
        validation_warnings=validation_warnings,
    )


def _check_no_cycle(ledger: TaskLedger) -> None:
    graph = {task.id: list(task.dependencies) for task in ledger.tasks}
    states: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> None:
        state = states.get(node, 0)
        if state == 2:
            return
        if state == 1:
            start = stack.index(node) if node in stack else 0
            cycle = " -> ".join(stack[start:] + [node])
            raise PlanError(f"Cycle de dépendances détecté : {cycle}.", semantic=True)
        states[node] = 1
        stack.append(node)
        for dependency in graph.get(node, []):
            if dependency not in graph:
                raise PlanError(f"Dépendance inconnue : {dependency}.")
            visit(dependency)
        stack.pop()
        states[node] = 2

    for task_id in graph:
        visit(task_id)
