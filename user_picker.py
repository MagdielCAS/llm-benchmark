"""
user_picker.py — Interactive session for manual "best answer" selection.

After automated scoring completes, this module presents the user with a
random sample of prompts (up to :data:`config.USER_PICKS` prompts) across
different categories.  For each prompt, all model answers are displayed
side-by-side in the terminal using :mod:`rich` panels, and the user picks
the one they find best via a :mod:`questionary` select menu.

Picks are persisted to ``results/user_picks.md`` via :mod:`progress`.  On a
resumed run, prompts that were already picked are automatically skipped.
"""

import random
from typing import Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import progress as prog
from config import ALL_PROMPTS, USER_PICKS

console = Console()


def _display_answers(prompt_obj, answers: dict[str, str]) -> None:
    """Pretty-print the question and all model answers for user review."""
    console.print(
        Panel(
            f"[bold yellow]Category:[/bold yellow] {prompt_obj.category}\n\n"
            f"[bold cyan]{prompt_obj.label}[/bold cyan]\n\n"
            + prompt_obj.user,
            title="📋 Prompt",
            border_style="blue",
        )
    )
    for model, answer in answers.items():
        console.print(
            Panel(
                answer,
                title=f"[bold magenta]{model}[/bold magenta]",
                border_style="magenta",
            )
        )


def run_user_picks(models: list[str]) -> None:
    """Interactively let the user pick the best answer for a sample of prompts."""
    console.print("\n[bold cyan]━━━ Phase 3: Your Manual Picks ━━━[/bold cyan]")
    console.print(
        f"[dim]You'll be shown answers for up to {USER_PICKS} prompts. "
        "Pick the one you find best.[/dim]\n"
    )

    # Load existing picks so we can skip already answered ones
    existing_picks = prog.load_user_picks()
    already_picked = {p["prompt_id"] for p in existing_picks}

    # Determine eligible prompt IDs (at least 2 models answered)
    eligible = []
    for prompt_id, prompt_obj in ALL_PROMPTS.items():
        if prompt_id in already_picked:
            continue
        answers = {
            m: prog.load_answer(m, prompt_id)
            for m in models
            if prog.is_answer_done(m, prompt_id)
        }
        answers = {m: a for m, a in answers.items() if a}
        if len(answers) >= 2:
            eligible.append((prompt_id, prompt_obj, answers))

    if not eligible:
        if existing_picks:
            console.print("[green]✓ All prompts already picked.[/green]\n")
        else:
            console.print("[yellow]⚠ No eligible prompts found for manual picks.[/yellow]\n")
        return

    # Sample randomly, up to USER_PICKS
    sample = random.sample(eligible, min(USER_PICKS, len(eligible)))
    picks = list(existing_picks)  # carry over previous picks

    for i, (prompt_id, prompt_obj, answers) in enumerate(sample, 1):
        console.rule(f"[bold]Pick {i} of {len(sample)}[/bold]")
        _display_answers(prompt_obj, answers)

        choices = list(answers.keys()) + ["⏭  Skip this prompt"]
        winner = questionary.select(
            f"Which answer is best for '{prompt_obj.label}'?",
            choices=choices,
        ).ask()

        if winner and not winner.startswith("⏭"):
            picks.append(
                {
                    "prompt_id": prompt_id,
                    "category": prompt_obj.category,
                    "winner": winner,
                }
            )
            console.print(f"  [green]✓ Recorded:[/green] [bold]{winner}[/bold]\n")
        else:
            console.print("  [dim]Skipped.[/dim]\n")

    prog.save_user_picks(picks)
    console.print(f"[bold green]✓ User picks saved ({len(picks)} total).[/bold green]\n")
