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
:func:`_extract_json` provides a best-effort fallback for models that wrap
their JSON response in markdown fences or prose.
"""

import json
import re
import random
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import ollama_client as ollama
import progress as prog
import validator as vld
from config import ALL_PROMPTS, ANONYMOUS_SCORING_SYSTEM, NAMED_SCORING_SYSTEM, FORMAT_CHECKED_PROMPT_IDS

console = Console()


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract and parse a JSON object from model output."""
    # Try direct parse first
    try:
        return json.loads(text)
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
    done_ops = 0

    console.print("\n[bold cyan]━━━ Phase 2: Scoring ━━━[/bold cyan]\n")

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
            console.print(
                f"  [yellow]⚠ Skipping scoring for [bold]{prompt_id}[/bold] — fewer than 2 models answered.[/yellow]"
            )
            done_ops += len(models) * 2
            continue

        user_prompt = prompt_obj.user

        for scoring_model in models:
            # ── Anonymous scoring ──
            if not prog.is_score_done(scoring_model, prompt_id, "anonymous"):
                anon_body, label_map = _build_anonymous_prompt(user_prompt, answers)
                console.print(
                    f"  [dim]Scoring[/dim] [bold]{prompt_id}[/bold] "
                    f"[dim]anonymously with[/dim] [cyan]{scoring_model}[/cyan]..."
                )
                try:
                    text, _, _ = ollama.generate(scoring_model, anon_body, ANONYMOUS_SCORING_SYSTEM)
                    parsed = _extract_json(text)
                    if parsed and "scores" in parsed:
                        # Re-map labels back to model names
                        named_scores = {label_map[lbl]: score for lbl, score in parsed["scores"].items() if lbl in label_map}
                        named_rationale = {label_map[lbl]: rat for lbl, rat in parsed.get("rationale", {}).items() if lbl in label_map}
                        prog.save_score(scoring_model, prompt_id, "anonymous", named_scores, named_rationale)
                        console.print(f"    [green]✓[/green] Anonymous scores: {named_scores}")
                    else:
                        console.print(f"    [yellow]⚠ Could not parse JSON from {scoring_model}. Raw: {text[:200]}[/yellow]")
                except Exception as e:
                    console.print(f"    [red]✗ Error: {e}[/red]")
            else:
                console.print(f"  [dim]↩ Skipping anonymous score {prompt_id} × {scoring_model} (already done)[/dim]")

            done_ops += 1

            # ── Named scoring ──
            if not prog.is_score_done(scoring_model, prompt_id, "named"):
                named_body = _build_named_prompt(user_prompt, answers)
                console.print(
                    f"  [dim]Scoring[/dim] [bold]{prompt_id}[/bold] "
                    f"[dim]with names using[/dim] [cyan]{scoring_model}[/cyan]..."
                )
                try:
                    text, _, _ = ollama.generate(scoring_model, named_body, NAMED_SCORING_SYSTEM)
                    parsed = _extract_json(text)
                    if parsed and "scores" in parsed:
                        prog.save_score(scoring_model, prompt_id, "named", parsed["scores"], parsed.get("rationale", {}))
                        console.print(f"    [green]✓[/green] Named scores: {parsed['scores']}")
                    else:
                        console.print(f"    [yellow]⚠ Could not parse JSON from {scoring_model}. Raw: {text[:200]}[/yellow]")
                except Exception as e:
                    console.print(f"    [red]✗ Error: {e}[/red]")
            else:
                console.print(f"  [dim]↩ Skipping named score {prompt_id} × {scoring_model} (already done)[/dim]")

            done_ops += 1

    console.print(f"\n[bold green]✓ Scoring complete.[/bold green] ({done_ops}/{total_ops} operations)\n")

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

