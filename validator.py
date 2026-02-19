"""
validator.py — Mechanical format compliance checker for structured output prompts.

For benchmark categories where the *format* of the answer is objectively verifiable
(``structured_output``, ``data_extraction``), this module provides deterministic
pass/fail checks that complement the LLM-based subjective scoring.

Each checker returns a :class:`ValidationResult` with:

* ``valid`` — whether the output is parseable and meets the schema constraints.
* ``score`` — a 0–10 integer derived from how many schema requirements were met.
* ``details`` — human-readable breakdown of what passed and what failed.

These mechanical scores are stored alongside LLM scores and shown in a dedicated
"Format Compliance" table in the final report.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class ValidationResult:
    """Result of a mechanical format validation check."""

    valid: bool
    """Whether the output fully satisfies the expected format."""

    score: int
    """0–10 compliance score (10 = perfect, 0 = completely wrong format)."""

    details: list[str] = field(default_factory=list)
    """List of individual check outcomes (passed or failed)."""

    parsed: Optional[Any] = None
    """The successfully parsed object, if any."""


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[dict]:
    """Try to extract and parse the first JSON object ``{...}`` from text."""
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _extract_json_array(text: str) -> Optional[list]:
    """Try to extract and parse the first JSON array ``[...]`` from text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def _extract_frontmatter(text: str) -> Optional[dict]:
    """Try to parse YAML frontmatter from a markdown string."""
    text = text.strip()
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None
    return None


# ---------------------------------------------------------------------------
# Per-prompt validators
# ---------------------------------------------------------------------------

def validate_struct_1(text: str) -> ValidationResult:
    """Validate JSON schema compliance for the Streamflow project prompt."""
    obj = _extract_json_object(text)
    if obj is None:
        return ValidationResult(valid=False, score=0, details=["❌ Response is not parseable JSON"])

    details = []
    score = 0
    checks = [
        ("name", lambda o: isinstance(o.get("name"), str) and o["name"]),
        ("version (semver)", lambda o: bool(re.match(r"^\d+\.\d+\.\d+$", str(o.get("version", ""))))),
        ("status enum", lambda o: o.get("status") in {"active", "archived", "experimental"}),
        ("tags (array of 3)", lambda o: isinstance(o.get("tags"), list) and len(o["tags"]) == 3),
        ("metadata.created_at", lambda o: isinstance(o.get("metadata", {}).get("created_at"), str)),
        ("metadata.stars (int)", lambda o: isinstance(o.get("metadata", {}).get("stars"), int)),
        ("metadata.license", lambda o: isinstance(o.get("metadata", {}).get("license"), str) and o["metadata"]["license"]),
        ("no extra top-level keys", lambda o: set(o.keys()) <= {"name", "version", "status", "tags", "metadata"}),
    ]

    for label, check in checks:
        try:
            passed = check(obj)
        except Exception:
            passed = False
        if passed:
            details.append(f"✅ {label}")
            score += 1
        else:
            details.append(f"❌ {label}")

    # Scale to 0–10
    score = round(score / len(checks) * 10)
    return ValidationResult(valid=score >= 8, score=score, details=details, parsed=obj)


def validate_struct_2(text: str) -> ValidationResult:
    """Validate YAML frontmatter markdown for the article prompt."""
    fm = _extract_frontmatter(text)
    if fm is None:
        return ValidationResult(valid=False, score=0, details=["❌ No valid YAML frontmatter found"])

    details = []
    score = 0
    checks = [
        ("title present", lambda f: isinstance(f.get("title"), str) and f["title"]),
        ("date (string)", lambda f: isinstance(f.get("date"), str) and re.search(r"\d{4}", str(f["date"]))),
        ("author present", lambda f: isinstance(f.get("author"), str) and f["author"]),
        ("tags (list of 3)", lambda f: isinstance(f.get("tags"), list) and len(f["tags"]) == 3),
        ("draft (boolean)", lambda f: isinstance(f.get("draft"), bool)),
        ("reading_time_min (int)", lambda f: isinstance(f.get("reading_time_min"), int)),
        ("markdown body present", lambda _: bool(re.search(r"---\s*\n.*\n---\s*\n.+", text, re.DOTALL))),
    ]

    for label, check in checks:
        try:
            passed = check(fm)
        except Exception:
            passed = False
        if passed:
            details.append(f"✅ {label}")
            score += 1
        else:
            details.append(f"❌ {label}")

    score = round(score / len(checks) * 10)
    return ValidationResult(valid=score >= 8, score=score, details=details, parsed=fm)


def validate_struct_3(text: str) -> ValidationResult:
    """Validate the flight booking function call JSON."""
    obj = _extract_json_object(text)
    if obj is None:
        return ValidationResult(valid=False, score=0, details=["❌ Response is not parseable JSON"])

    details = []
    score = 0
    params = obj.get("parameters", {})
    checks = [
        ("function = book_flight", lambda o: o.get("function") == "book_flight"),
        ("parameters present", lambda o: isinstance(o.get("parameters"), dict)),
        ("origin (IATA ~3 chars)", lambda o: isinstance(params.get("origin"), str) and 2 <= len(params["origin"]) <= 4),
        ("destination (IATA)", lambda o: isinstance(params.get("destination"), str) and 2 <= len(params["destination"]) <= 4),
        ("departure_date (YYYY-MM-DD)", lambda o: bool(re.match(r"\d{4}-\d{2}-\d{2}", str(params.get("departure_date", ""))))),
        ("passengers.adults (int)", lambda o: isinstance(params.get("passengers", {}).get("adults"), int)),
        ("passengers.children (int)", lambda o: isinstance(params.get("passengers", {}).get("children"), int)),
        ("cabin_class enum", lambda o: params.get("cabin_class") in {"economy", "business", "first"}),
        ("seat_preference enum", lambda o: params.get("seat_preference") in {"window", "aisle", "middle", "none"}),
    ]

    for label, check in checks:
        try:
            passed = check(obj)
        except Exception:
            passed = False
        if passed:
            details.append(f"✅ {label}")
            score += 1
        else:
            details.append(f"❌ {label}")

    score = round(score / len(checks) * 10)
    return ValidationResult(valid=score >= 8, score=score, details=details, parsed=obj)


def validate_extract_1(text: str) -> ValidationResult:
    """Validate support ticket extraction → JSON with ENUMs."""
    obj = _extract_json_object(text)
    if obj is None:
        return ValidationResult(valid=False, score=0, details=["❌ Response is not parseable JSON"])

    details = []
    score = 0
    checks = [
        ("ticket_id present", lambda o: isinstance(o.get("ticket_id"), str) and o["ticket_id"]),
        ("priority enum", lambda o: o.get("priority") in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}),
        ("category enum", lambda o: o.get("category") in {"BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING", "OTHER"}),
        ("sentiment enum", lambda o: o.get("sentiment") in {"POSITIVE", "NEUTRAL", "NEGATIVE", "ANGRY"}),
        ("summary (str, ≤20 words)", lambda o: isinstance(o.get("summary"), str) and len(o["summary"].split()) <= 20),
        ("requires_human (bool)", lambda o: isinstance(o.get("requires_human"), bool)),
        ("mentioned_products (list)", lambda o: isinstance(o.get("mentioned_products"), list)),
        # Spot-check: ticket has both billing AND technical issues
        ("catches billing issue", lambda o: o.get("category") in {"BILLING", "TECHNICAL"}),
        ("correct high/critical priority", lambda o: o.get("priority") in {"HIGH", "CRITICAL"}),
    ]

    for label, check in checks:
        try:
            passed = check(obj)
        except Exception:
            passed = False
        if passed:
            details.append(f"✅ {label}")
            score += 1
        else:
            details.append(f"❌ {label}")

    score = round(score / len(checks) * 10)
    return ValidationResult(valid=score >= 7, score=score, details=details, parsed=obj)


def validate_extract_2(text: str) -> ValidationResult:
    """Validate corrupted JSON repair and enum normalisation."""
    obj = _extract_json_object(text)
    if obj is None:
        return ValidationResult(valid=False, score=0, details=["❌ Response is not parseable JSON"])

    details = []
    score = 0
    loc = obj.get("location", {})
    lr = obj.get("last_reading", {})
    checks = [
        ("id present", lambda o: isinstance(o.get("id"), str) and o["id"]),
        ("type enum", lambda o: o.get("type") in {"SENSOR", "ACTUATOR", "GATEWAY", "CONTROLLER"}),
        ("status enum", lambda o: o.get("status") in {"ONLINE", "OFFLINE", "DEGRADED", "UNKNOWN"}),
        ("location.building (str)", lambda o: isinstance(loc.get("building"), str) and loc["building"]),
        ("location.floor (int)", lambda o: isinstance(loc.get("floor"), int)),
        ("location.room (str)", lambda o: isinstance(loc.get("room"), str) and loc["room"]),
        ("last_reading.value (number)", lambda o: isinstance(lr.get("value"), (int, float))),
        ("last_reading.unit (str)", lambda o: isinstance(lr.get("unit"), str) and lr["unit"]),
        ("last_reading.timestamp (ISO-8601)", lambda o: bool(re.search(r"\d{4}-\d{2}-\d{2}", str(lr.get("timestamp", ""))))),
        ("alerts (list)", lambda o: isinstance(o.get("alerts"), list)),
    ]

    for label, check in checks:
        try:
            passed = check(obj)
        except Exception:
            passed = False
        if passed:
            details.append(f"✅ {label}")
            score += 1
        else:
            details.append(f"❌ {label}")

    score = round(score / len(checks) * 10)
    return ValidationResult(valid=score >= 8, score=score, details=details, parsed=obj)


def validate_extract_3(text: str) -> ValidationResult:
    """Validate markdown table → JSON array with enum normalisation."""
    arr = _extract_json_array(text)
    if arr is None:
        return ValidationResult(valid=False, score=0, details=["❌ Response is not a parseable JSON array"])

    details = []
    score = 0
    valid_roles = {"ENGINEER", "DESIGNER", "MANAGER", "QA", "DEVOPS"}
    valid_levels = {"JUNIOR", "MID", "SENIOR", "LEAD", "PRINCIPAL"}

    checks = [
        ("returns a list", lambda a: isinstance(a, list)),
        ("5 entries (one per row)", lambda a: len(a) == 5),
        ("all entries have 'name'", lambda a: all(isinstance(e.get("name"), str) for e in a)),
        ("all roles are valid enum", lambda a: all(e.get("role") in valid_roles for e in a)),
        ("all levels are valid enum", lambda a: all(e.get("level") in valid_levels for e in a)),
        ("'active' is boolean everywhere", lambda a: all(isinstance(e.get("active"), bool) for e in a)),
        ("team field present on all", lambda a: all(isinstance(e.get("team"), str) for e in a)),
        # Spot-check specific rows
        ("Ana = ENGINEER SENIOR active", lambda a: any(
            e.get("name", "").startswith("Ana") and e.get("role") == "ENGINEER"
            and e.get("level") == "SENIOR" and e.get("active") is True
            for e in a
        )),
        ("Eva = DEVOPS PRINCIPAL inactive", lambda a: any(
            e.get("name", "").startswith("Eva") and e.get("role") == "DEVOPS"
            and e.get("active") is False
            for e in a
        )),
    ]

    for label, check in checks:
        try:
            passed = check(arr)
        except Exception:
            passed = False
        if passed:
            details.append(f"✅ {label}")
            score += 1
        else:
            details.append(f"❌ {label}")

    score = round(score / len(checks) * 10)
    return ValidationResult(valid=score >= 7, score=score, details=details, parsed=arr)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_VALIDATORS = {
    "struct_1": validate_struct_1,
    "struct_2": validate_struct_2,
    "struct_3": validate_struct_3,
    "extract_1": validate_extract_1,
    "extract_2": validate_extract_2,
    "extract_3": validate_extract_3,
}


def validate(prompt_id: str, answer_text: str) -> Optional[ValidationResult]:
    """
    Run the mechanical validator for a given prompt, if one exists.

    Parameters
    ----------
    prompt_id:
        The prompt identifier (e.g. ``'struct_1'``, ``'extract_2'``).
    answer_text:
        The raw model output to validate.

    Returns
    -------
    ValidationResult or None
        ``None`` if no mechanical validator exists for this prompt.
    """
    fn = _VALIDATORS.get(prompt_id)
    if fn is None:
        return None
    return fn(answer_text)
