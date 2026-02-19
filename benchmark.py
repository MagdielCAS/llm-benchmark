#!/usr/bin/env python3
"""
benchmark.py — Main entry-point for the LLM benchmark suite.

Usage:
    python benchmark.py

Phases:
  1. Answer generation — each selected model answers all prompts
  2. Scoring           — each model judges all answers (anonymous + named)
  3. User picks        — you manually pick the best answer for a sample
  4. Report            — results/report.md is generated
"""

import sys
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import ollama_client as ollama
import progress as prog
import scorer
import user_picker
import reporter
from config import CATEGORIES, ALL_PROMPTS

console = Console()

BANNER = """
[bold cyan]
 ██╗     ██╗     ███╗   ███╗    ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
 ██║     ██║     ████╗ ████║    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
 ██║     ██║     ██╔████╔██║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
 ██║     ██║     ██║╚██╔╝██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
 ███████╗███████╗██║ ╚═╝ ██║    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
 ╚══════╝╚══════╝╚═╝     ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
[/bold cyan]
[dim]   Local LLM Benchmark Suite · Powered by Ollama · M1 MacBook Pro Edition[/dim]
"""


def check_ollama() -> None:
    """Verify Ollama is running, exit with a helpful message if not."""
    if not ollama.check_running():
        console.print(
            Panel(
                "[bold red]Ollama is not running.[/bold red]\n\n"
                "Please start it with:  [cyan]ollama serve[/cyan]\n"
                "Then re-run this script.",
                title="❌ Connection Error",
                border_style="red",
            )
        )
        sys.exit(1)


def select_models() -> list[str]:
    """Prompt the user to choose which local models to benchmark."""
    available = ollama.list_models()
    if not available:
        console.print("[red]No local models found. Pull at least one model with: ollama pull <name>[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Found {len(available)} local model(s):[/bold]")

    selected = questionary.checkbox(
        "Select models to benchmark (space to toggle, enter to confirm):",
        choices=available,
    ).ask()

    if not selected:
        console.print("[yellow]No models selected. Exiting.[/yellow]")
        sys.exit(0)

    console.print(f"\n[green]✓ Benchmarking:[/green] {', '.join(selected)}\n")
    return selected


def ask_resume_or_restart() -> bool:
    """
    If previous results exist, ask user whether to resume or start fresh.
    Returns True if resuming, False if restarting.
    """
    if not prog.has_any_results():
        return False

    choice = questionary.select(
        "Previous benchmark results found. What would you like to do?",
        choices=[
            "▶  Resume from where I left off",
            "🔄  Start over (delete all previous results)",
        ],
    ).ask()

    if choice and choice.startswith("🔄"):
        import shutil
        from pathlib import Path

        results_dir = Path(__file__).parent / "results"
        shutil.rmtree(results_dir, ignore_errors=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "answers").mkdir(exist_ok=True)
        (results_dir / "scores").mkdir(exist_ok=True)
        console.print("[yellow]🗑  Previous results cleared. Starting fresh.[/yellow]\n")
        return False

    console.print("[cyan]▶  Resuming previous run.[/cyan]\n")
    return True


def run_answer_phase(models: list[str]) -> None:
    """Phase 1: Generate answers for all model × prompt combinations."""
    console.print("[bold cyan]━━━ Phase 1: Generating Answers ━━━[/bold cyan]\n")

    total = len(models) * len(ALL_PROMPTS)
    done = sum(
        1
        for m in models
        for pid in ALL_PROMPTS
        if prog.is_answer_done(m, pid)
    )

    console.print(f"  Progress so far: {done}/{total} answers done.\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task("Generating answers…", total=total, completed=done)

        for model in models:
            for cat in CATEGORIES:
                for prompt_obj in cat.prompts:
                    pid = prompt_obj.id

                    if prog.is_answer_done(model, pid):
                        continue

                    progress_bar.update(
                        task_id,
                        description=f"[cyan]{model}[/cyan] → [bold]{prompt_obj.label}[/bold]",
                    )

                    try:
                        text, tokens, duration = ollama.generate(
                            model,
                            prompt_obj.user,
                            prompt_obj.system,
                        )
                        prog.save_answer(
                            model=model,
                            prompt_id=pid,
                            category=cat.id,
                            label=prompt_obj.label,
                            user_prompt=prompt_obj.user,
                            answer=text,
                            tokens=tokens,
                            duration_s=duration,
                        )
                        progress_bar.advance(task_id)
                    except Exception as e:
                        progress_bar.console.print(
                            f"  [red]✗ Error ({model} / {pid}): {e}[/red]"
                        )
                        # Save a placeholder so we can retry on next run
                        # (don't write status=done — it will be re-attempted)

    console.print("[bold green]✓ Answer generation complete.[/bold green]\n")


def main() -> None:
    console.print(BANNER)

    # ── Pre-flight checks ──
    check_ollama()

    # ── Model selection ──
    models = select_models()

    # ── Resume / restart ──
    ask_resume_or_restart()

    # ── Phase 1: Answers ──
    run_answer_phase(models)

    # ── Phase 2: Scoring ──
    scorer.run_scoring(models)

    # ── Phase 3: User picks ──
    user_picker.run_user_picks(models)

    # ── Phase 4: Report ──
    reporter.generate_report(models)

    console.print(
        Panel(
            "[bold green]🎉 Benchmark complete![/bold green]\n\n"
            "Results are stored in:  [cyan]results/[/cyan]\n"
            "Final report:          [cyan]results/report.md[/cyan]",
            title="✅ Done",
            border_style="green",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(
            "\n\n[yellow]⚠ Interrupted by user.[/yellow] "
            "Progress has been saved. Re-run and choose [bold]Resume[/bold] to continue.\n"
        )
        sys.exit(0)
