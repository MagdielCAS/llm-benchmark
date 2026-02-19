"""
scorer.py — Two-pass scoring of model answers using peer models as judges.

Scoring runs in two passes per (scoring_model, prompt_id) combination:

1. **Anonymous pass** — answers are shuffled and labelled A, B, C … so the
   scoring model has no idea which model produced which answer (blind review).
   Labels are then re-mapped back to model names before saving.

2. **Named pass** — answers are presented with model names visible.  The delta
   between named and anonymous scores is used as a self-bias indicator in the
   final report.

Each (scoring_model, prompt_id, mode) triple writes exactly one score file via
:mod:`progress`.  Pairs that already have a file are silently skipped,
enabling crash-safe resumption.

The model is expected to respond with valid JSON matching the schema defined
in :data:`config.ANONYMOUS_SCORING_SYSTEM` / :data:`config.NAMED_SCORING_SYSTEM`.
:func:`_extract_scores` provides a best-effort fallback for models that wrap
their JSON response in markdown fences or prose.
"""

import json
import re
import random
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
)

import ollama_client as ollama
import progress as prog
import validator as vld
from config import ALL_PROMPTS, ANONYMOUS_SCORING_SYSTEM, NAMED_SCORING_SYSTEM, FORMAT_CHECKED_PROMPT_IDS

console = Console()


def _parse_json_from_text(text: str) -> Optional[dict | list]:
    """Extract and parse the first JSON value from a model response.

    Handles three common wrapping patterns:
    1. Bare JSON (ideal case).
    2. JSON wrapped inside a markdown code fence (\\`\\`\\`json … \\`\\`\\`).
    3. Embedded JSON — finds the first ``{…}`` or ``[…]`` block in prose.
    """
    text = text.strip()

    # Strip markdown code fences if present
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```\s*$", "", fenced).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# Score labels that count as valid answer IDs
_ANSWER_LABELS = set("ABCDEFGHIJKLMNOP")


def _normalise_scores(raw: dict) -> Optional[dict]:
    """Normalise any observed scoring response shape into ``{scores, rationale}``.

    Models frequently ignore the exact JSON schema in the system prompt and
    instead produce one of the following observed formats:

    1. **Canonical** (pass-through)::

           {"scores": {"A": 8, "B": 6}, "rationale": {"A": "…", "B": "…"}}

    2. **Responses array** (gemma3, some llama variants)::

           {"responses": [{"answer_id": "A", "score": 8, "reason": "…"}, …]}

       Also handles ``id`` / ``answer`` / ``label`` as the key field, and
       ``rationale`` / ``explanation`` / ``feedback`` as the reason field.

    3. **Per-label dict**::

           {"A": {"score": 8, "rationale": "…"}, "B": {"score": 6, "rationale": "…"}}

    4. **Flat label→score map** (no wrapper key)::

           {"A": 8, "B": 6}

    5. **Nested ``evaluations`` / ``results`` / ``answers`` key** — same as #2
       but with a different container name.
    """
    if not isinstance(raw, dict):
        return None

    # ── 1. Canonical ──
    if "scores" in raw and isinstance(raw["scores"], dict):
        return raw

    # ── 2 & 5. Array under any plausible container key ──
    _ARRAY_KEYS = ("responses", "evaluations", "results", "answers", "items")
    for key in _ARRAY_KEYS:
        arr = raw.get(key)
        if not isinstance(arr, list):
            continue
        scores: dict[str, int] = {}
        rationale: dict[str, str] = {}
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            # find the label field
            label = None
            for lk in ("answer_id", "id", "label", "answer", "letter", "key"):
                val = entry.get(lk)
                if isinstance(val, str) and val.upper() in _ANSWER_LABELS:
                    label = val.upper()
                    break
            if label is None:
                continue
            # find score
            for sk in ("score", "rating", "points", "value"):
                s = entry.get(sk)
                if isinstance(s, (int, float)):
                    scores[label] = int(round(s))
                    break
            # find rationale
            for rk in ("reason", "rationale", "explanation", "feedback", "comment", "text"):
                r = entry.get(rk)
                if isinstance(r, str):
                    rationale[label] = r
                    break
        if scores:
            return {"scores": scores, "rationale": rationale}

    # ── 3. Per-label dict: {"A": {"score": 8, "rationale": "…"}} ──
    scores = {}
    rationale = {}
    all_labels = True
    for k, v in raw.items():
        if k.upper() not in _ANSWER_LABELS:
            all_labels = False
            break
        if isinstance(v, dict):
            for sk in ("score", "rating", "points", "value"):
                s = v.get(sk)
                if isinstance(s, (int, float)):
                    scores[k.upper()] = int(round(s))
                    break
            for rk in ("rationale", "reason", "explanation", "feedback"):
                r = v.get(rk)
                if isinstance(r, str):
                    rationale[k.upper()] = r
                    break
        elif isinstance(v, (int, float)):
            # ── 4. Flat label→score map ──
            scores[k.upper()] = int(round(v))
    if all_labels and scores:
        return {"scores": scores, "rationale": rationale}

    return None


def _extract_scores(text: str) -> Optional[dict]:
    """Parse a model's scoring response and return a normalised ``{scores, rationale}`` dict."""
    raw = _parse_json_from_text(text)
    if raw is None:
        return None
    return _normalise_scores(raw)


def _build_anonymous_prompt(user_prompt: str, answers: dict[str, str]) -> str:
    """Format the scoring prompt with labelled-but-anonymous answers."""
    labels = list("ABCDEFGHIJKLMNOP")
    items = list(answers.items())
    random.shuffle(items)

    label_map: dict[str, str] = {}  # label -> model_name
    sections = [f"## Original Question\n\n{user_prompt}\n\n## Answers to Score\n"]
    for i, (model, answer) in enumerate(items):
        lbl = labels[i]
        label_map[lbl] = model
        sections.append(f"### Answer {lbl}\n\n{answer}\n")

    return "\n".join(sections), label_map


def _build_named_prompt(user_prompt: str, answers: dict[str, str]) -> str:
    """Format the scoring prompt with named answers."""
    sections = [f"## Original Question\n\n{user_prompt}\n\n## Answers to Score\n"]
    for model, answer in answers.items():
        sections.append(f"### {model}\n\n{answer}\n")
    return "\n".join(sections)


def run_scoring(models: list[str]) -> None:
    """Run both anonymous and named scoring for all (scoring_model, prompt_id) pairs."""
    prompt_ids = list(ALL_PROMPTS.keys())

    total_ops = len(models) * len(prompt_ids) * 2  # anon + named per model per prompt
    phase_start = time.monotonic()

    console.print("\n[bold cyan]━━━ Phase 2: Scoring ━━━[/bold cyan]")
    console.print(
        f"  [dim]{len(models)} models × {len(prompt_ids)} prompts × 2 passes "
        f"= [bold]{total_ops}[/bold] scoring calls[/dim]\n"
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("Scoring", total=total_ops)

        # Track parse outcomes per model for the success-rate report section
        parse_ok: dict[str, dict[str, int]] = {m: {"anon": 0, "named": 0} for m in models}
        parse_fail: dict[str, dict[str, list]] = {m: {"anon": [], "named": []} for m in models}

        for prompt_id in prompt_ids:
            prompt_obj = ALL_PROMPTS[prompt_id]

            # Collect all available answers for this prompt
            answers: dict[str, str] = {}
            for m in models:
                if prog.is_answer_done(m, prompt_id):
                    ans = prog.load_answer(m, prompt_id)
                    if ans:
                        answers[m] = ans

            if len(answers) < 2:
                progress.console.print(
                    f"  [yellow]⚠ Skipping {prompt_id} — fewer than 2 models answered.[/yellow]"
                )
                progress.advance(task, len(models) * 2)
                continue

            user_prompt = prompt_obj.user

            for scoring_model in models:
                # ── Anonymous scoring ──
                progress.update(task, description=f"[anon]  {prompt_id} / {scoring_model}")
                if not prog.is_score_done(scoring_model, prompt_id, "anonymous"):
                    anon_body, label_map = _build_anonymous_prompt(user_prompt, answers)
                    t0 = time.monotonic()
                    try:
                        text, _, _ = ollama.generate(scoring_model, anon_body, ANONYMOUS_SCORING_SYSTEM)
                        elapsed = time.monotonic() - t0
                        parsed = _extract_scores(text)
                        if parsed:
                            named_scores = {label_map[lbl]: score for lbl, score in parsed["scores"].items() if lbl in label_map}
                            named_rationale = {label_map[lbl]: rat for lbl, rat in parsed.get("rationale", {}).items() if lbl in label_map}
                            prog.save_score(scoring_model, prompt_id, "anonymous", named_scores, named_rationale)
                            progress.console.print(
                                f"  [green]✓[/green] anon  [bold]{prompt_id}[/bold] "
                                f"← [cyan]{scoring_model}[/cyan]  "
                                f"[dim]scores={list(named_scores.values())}  {elapsed:.1f}s[/dim]"
                            )
                            parse_ok[scoring_model]["anon"] += 1
                        else:
                            progress.console.print(
                                f"  [yellow]⚠ anon {prompt_id} ← {scoring_model}: unrecognised JSON format[/yellow]  "
                                f"[dim]{text[:160]}[/dim]"
                            )
                            parse_fail[scoring_model]["anon"].append(prompt_id)
                    except Exception as e:
                        progress.console.print(f"  [red]✗ anon {prompt_id} ← {scoring_model}: {e}[/red]")
                        parse_fail[scoring_model]["anon"].append(prompt_id)
                else:
                    progress.console.print(
                        f"  [dim]↩ skip anon   {prompt_id} / {scoring_model}[/dim]"
                    )
                    parse_ok[scoring_model]["anon"] += 1  # already saved = was parseable

                progress.advance(task, 1)

                progress.update(task, description=f"[named] {prompt_id} / {scoring_model}")
                if not prog.is_score_done(scoring_model, prompt_id, "named"):
                    named_body = _build_named_prompt(user_prompt, answers)
                    t0 = time.monotonic()
                    try:
                        text, _, _ = ollama.generate(scoring_model, named_body, NAMED_SCORING_SYSTEM)
                        elapsed = time.monotonic() - t0
                        parsed = _extract_scores(text)
                        if parsed:
                            prog.save_score(scoring_model, prompt_id, "named", parsed["scores"], parsed.get("rationale", {}))
                            progress.console.print(
                                f"  [green]✓[/green] named [bold]{prompt_id}[/bold] "
                                f"← [cyan]{scoring_model}[/cyan]  "
                                f"[dim]scores={list(parsed['scores'].values())}  {elapsed:.1f}s[/dim]"
                            )
                            parse_ok[scoring_model]["named"] += 1
                        else:
                            progress.console.print(
                                f"  [yellow]⚠ named {prompt_id} ← {scoring_model}: unrecognised JSON format[/yellow]  "
                                f"[dim]{text[:160]}[/dim]"
                            )
                            parse_fail[scoring_model]["named"].append(prompt_id)
                    except Exception as e:
                        progress.console.print(f"  [red]✗ named {prompt_id} ← {scoring_model}: {e}[/red]")
                        parse_fail[scoring_model]["named"].append(prompt_id)
                else:
                    progress.console.print(
                        f"  [dim]↩ skip named  {prompt_id} / {scoring_model}[/dim]"
                    )
                    parse_ok[scoring_model]["named"] += 1  # already saved = was parseable

                progress.advance(task, 1)

    total_elapsed = time.monotonic() - phase_start
    mins, secs = divmod(int(total_elapsed), 60)
    console.print(
        f"\n[bold green]✓ Scoring complete[/bold green]  "
        f"[dim]total time: {mins}m {secs:02d}s[/dim]\n"
    )

    # Persist parse success/failure stats so the reporter can surface them
    import json as _json
    stats_path = Path(__file__).parent / "results" / "scoring_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    scoring_stats = {}
    for model in models:
        anon_ok = parse_ok[model]["anon"]
        named_ok = parse_ok[model]["named"]
        anon_fail = parse_fail[model]["anon"]
        named_fail = parse_fail[model]["named"]
        total = anon_ok + len(anon_fail) + named_ok + len(named_fail)
        success = anon_ok + named_ok
        scoring_stats[model] = {
            "anon_ok": anon_ok,
            "named_ok": named_ok,
            "anon_fail": anon_fail,
            "named_fail": named_fail,
            "total_attempts": total,
            "total_success": success,
            "success_rate": round(success / total, 3) if total else 1.0,
        }
    stats_path.write_text(_json.dumps(scoring_stats, indent=2), encoding="utf-8")
    console.print(f"[dim]  Scoring stats saved to {stats_path}[/dim]\n")

    # ── Phase 2b: Mechanical format validation ──
    _run_mechanical_validation(models)


def _run_mechanical_validation(models: list[str]) -> None:
    """Run deterministic format validators for structured/extraction prompts.

    For each model × format-checked prompt, a ``mode='mechanical'`` score file
    is written containing the 0–10 compliance score and per-field pass/fail details.
    These are completely independent of LLM-based scoring and serve as objective
    ground truth for the Format Compliance section of the report.
    """
    fmt_prompts = [pid for pid in ALL_PROMPTS if pid in FORMAT_CHECKED_PROMPT_IDS]
    if not fmt_prompts:
        return

    console.print("[bold cyan]━━━ Phase 2b: Mechanical Format Validation ━━━[/bold cyan]\n")

    for model in models:
        for prompt_id in fmt_prompts:
            if prog.is_score_done(model, prompt_id, "mechanical"):
                console.print(f"  [dim]↩ Skipping mechanical {prompt_id} × {model} (already done)[/dim]")
                continue

            answer = prog.load_answer(model, prompt_id)
            if not answer:
                continue

            result = vld.validate(prompt_id, answer)
            if result is None:
                continue

            prog.save_score(
                scoring_model=model,
                prompt_id=prompt_id,
                mode="mechanical",
                scores={model: result.score},
                rationale={model: " | ".join(result.details)},
            )
            status = "[green]✓[/green]" if result.valid else "[red]✗[/red]"
            console.print(
                f"  {status} {model} / {prompt_id} "
                f"— mechanical score: [bold]{result.score}/10[/bold]  "
                f"({'valid' if result.valid else 'invalid'})"
            )

    console.print("[bold green]✓ Mechanical validation complete.[/bold green]\n")

