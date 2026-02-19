"""
reporter.py — Aggregate all benchmark data and write the final markdown report.

The report (``results/report.md``) contains:

* **Overall leaderboard** — average anonymous score, average named score, and
  the bias delta (named − anonymous) per model, sorted by anonymous score.
* **Per-category anonymous scores** — table of average scores broken down by
  the 8 benchmark categories.
* **Per-category named scores** — same table but using the named-scoring pass.
* **User picks** — the prompts the user manually evaluated and their winners,
  plus a simple tally of how many times each model was chosen.
* **Per-model appendix** — individual prompt-level scores for quick reference.

Data is read from the ``results/answers/`` and ``results/scores/`` directories
via :mod:`progress`.  The report is always regenerated from scratch on each
call so it reflects the latest state of all saved files.
"""

from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from rich.console import Console

import progress as prog
from config import ALL_PROMPTS, CATEGORIES

console = Console()

REPORT_PATH = Path(__file__).parent / "results" / "report.md"


def _avg(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{sum(values) / len(values):.2f}"


def generate_report(models: list[str]) -> None:
    """Collect all scores and user picks, then write results/report.md."""
    console.print("\n[bold cyan]━━━ Phase 4: Generating Report ━━━[/bold cyan]\n")

    # ── Collect scores ──
    all_scores = prog.get_all_scores()

    # anon_scores[model][prompt_id] = score
    anon_scores: dict[str, dict[str, int]] = defaultdict(dict)
    named_scores: dict[str, dict[str, int]] = defaultdict(dict)

    for s in all_scores:
        mode = s.get("mode", "")
        scores_map = s.get("scores", {})
        prompt_id = s.get("prompt_id", "")
        if mode == "anonymous":
            for model, score in scores_map.items():
                anon_scores[model][prompt_id] = score
        elif mode == "named":
            for model, score in scores_map.items():
                named_scores[model][prompt_id] = score

    # ── Collect user picks ──
    user_picks = prog.load_user_picks()

    # ── Build category → prompt_ids mapping ──
    cat_prompt_map: dict[str, list[str]] = defaultdict(list)
    for cat in CATEGORIES:
        for p in cat.prompts:
            cat_prompt_map[cat.id].append(p.id)

    lines: list[str] = []

    # ── Header ──
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        "# LLM Benchmark Report\n",
        f"> Generated: {now}  \n",
        f"> Models: {', '.join(f'`{m}`' for m in models)}  \n",
        f"> Total prompts: {len(ALL_PROMPTS)}  \n",
        "\n---\n",
    ]

    # ── Overall Leaderboard ──
    lines += ["## 🏆 Overall Leaderboard\n\n"]
    lines += ["| Rank | Model | Avg Anon Score | Avg Named Score | Bias (Named−Anon) |\n"]
    lines += ["|------|-------|---------------|-----------------|-------------------|\n"]

    leaderboard = []
    for model in models:
        anon_vals = list(anon_scores[model].values())
        named_vals = list(named_scores[model].values())
        avg_anon = sum(anon_vals) / len(anon_vals) if anon_vals else 0
        avg_named = sum(named_vals) / len(named_vals) if named_vals else 0
        bias = avg_named - avg_anon
        leaderboard.append((model, avg_anon, avg_named, bias))

    leaderboard.sort(key=lambda x: x[1], reverse=True)
    for rank, (model, avg_anon, avg_named, bias) in enumerate(leaderboard, 1):
        bias_str = f"+{bias:.2f}" if bias >= 0 else f"{bias:.2f}"
        lines.append(
            f"| {rank} | `{model}` | {avg_anon:.2f} | {avg_named:.2f} | {bias_str} |\n"
        )

    lines.append("\n> **Bias** = how much a model scores itself/others differently when it knows the names.\n\n")

    # ── Per-Category Breakdown ──
    lines += ["---\n\n## 📊 Per-Category Scores (Anonymous)\n\n"]
    header = "| Category | " + " | ".join(f"`{m}`" for m in models) + " |\n"
    sep = "|----------|" + "|".join(["------"] * len(models)) + "|\n"
    lines += [header, sep]

    for cat in CATEGORIES:
        row_vals = []
        for model in models:
            cat_scores = [anon_scores[model].get(pid, None) for pid in cat_prompt_map[cat.id]]
            cat_scores = [s for s in cat_scores if s is not None]
            row_vals.append(_avg(cat_scores))
        lines.append(f"| {cat.name} | " + " | ".join(row_vals) + " |\n")

    lines.append("\n")

    # ── Named Scores Table ──
    lines += ["---\n\n## 📊 Per-Category Scores (Named — includes self-awareness)\n\n"]
    lines += [header, sep]

    for cat in CATEGORIES:
        row_vals = []
        for model in models:
            cat_scores = [named_scores[model].get(pid, None) for pid in cat_prompt_map[cat.id]]
            cat_scores = [s for s in cat_scores if s is not None]
            row_vals.append(_avg(cat_scores))
        lines.append(f"| {cat.name} | " + " | ".join(row_vals) + " |\n")

    lines.append("\n")

    # ── User Picks ──
    lines += ["---\n\n## 👤 Your Manual Picks\n\n"]
    if user_picks:
        lines += ["| Prompt ID | Category | Your Winner |\n"]
        lines += ["|-----------|----------|-------------|\n"]
        pick_wins: dict[str, int] = defaultdict(int)
        for pick in user_picks:
            lines.append(f"| `{pick['prompt_id']}` | {pick['category']} | `{pick['winner']}` |\n")
            pick_wins[pick["winner"]] += 1

        lines.append("\n**Pick tally:**\n\n")
        for model, count in sorted(pick_wins.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- `{model}`: {count} win(s)\n")
    else:
        lines.append("_No manual picks recorded._\n")

    lines.append("\n")

    # ── Per-Model Answer Appendix ──
    lines += ["---\n\n## 📎 Per-Model Answer Summary\n\n"]
    for model in models:
        lines.append(f"### `{model}`\n\n")
        for cat in CATEGORIES:
            lines.append(f"#### {cat.name}\n\n")
            for p in cat.prompts:
                anon = anon_scores[model].get(p.id)
                named = named_scores[model].get(p.id)
                lines.append(f"- **{p.label}** (`{p.id}`) — Anon: {anon or '—'} / Named: {named or '—'}\n")
            lines.append("\n")

    # ── Write file ──
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    console.print(f"[bold green]✓ Report written to:[/bold green] {REPORT_PATH}\n")
