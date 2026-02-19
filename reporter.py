"""
reporter.py — Aggregate all benchmark data and write the final markdown report.

The report (``results/report.md``) is designed to support a data-driven
decision about which local LLM to use day-to-day.  It covers:

Quality
~~~~~~~
* **Overall leaderboard** — avg anonymous score, avg named score, and the
  self-bias delta per model.
* **Per-category anonymous scores** — 8-category breakdown (anonymous judging).
* **Per-category named scores** — same with model names visible.

Performance
~~~~~~~~~~~
* **Speed & throughput** — avg response time (s), tokens generated, and
  tokens/second (throughput) per model.
* **Latency heatmap** — per-prompt response time for every model (helps spot
  which categories are slow on which models).

Efficiency
~~~~~~~~~~
* **Quality-efficiency frontier** — combines score and throughput into a single
  composite "efficiency" metric: ``score × tok/s`` (quality per unit time).
* **Cost proxy** — total token output (proxy for energy/compute cost).

Human signal
~~~~~~~~~~~~
* **Your manual picks** — which model you found best on sampled prompts.
* **Per-model answer appendix** — individual prompt scores for deep-dive.

Data is read from the ``results/answers/`` and ``results/scores/``
directories via :mod:`progress`.  The report is fully regenerated on
each call so it always reflects the latest state of all saved files.
"""

from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import statistics

from rich.console import Console

import progress as prog
from config import ALL_PROMPTS, CATEGORIES, FORMAT_CHECKED_PROMPT_IDS, EXPORT_PDF

try:
    from markdown_pdf import Section, MarkdownPdf
except ImportError:
    MarkdownPdf = None

console = Console()

REPORT_PATH = Path(__file__).parent / "results" / "report.md"


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _mermaid_bar(
    title: str,
    x_labels: list[str],
    values: list[float],
    y_label: str = "",
    y_max: Optional[float] = None,
) -> str:
    """Render a Mermaid xychart-beta bar chart block.

    Model names with colons (e.g. ``llama3.2:3b``) are sanitised to avoid
    Mermaid parse errors — colons are replaced with a middle dot ·.
    """
    safe_labels = [l.replace(":", "\u00b7").replace('"', '') for l in x_labels]
    label_str = ", ".join(f'"{l}"' for l in safe_labels)
    val_str = ", ".join(f"{v:.2f}" for v in values)
    max_line = f"    y-axis \"{y_label}\" 0 --> {y_max}" if y_max else f"    y-axis \"{y_label}\""
    return (
        f"```mermaid\n"
        f"xychart-beta\n"
        f'    title "{title}"\n'
        f"    x-axis [{label_str}]\n"
        f"{max_line}\n"
        f"    bar [{val_str}]\n"
        f"```\n"
    )


def _ascii_bar(label: str, value: float, max_val: float, width: int = 20) -> str:
    """Single-line ASCII bar for terminal-friendly fallback."""
    filled = int(round(value / max_val * width)) if max_val else 0
    bar = "█" * filled + "░" * (width - filled)
    return f"`{bar}` {value:.2f}"



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{sum(values) / len(values):.2f}"


def _fmt(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    """Format a numeric value or return em-dash if None."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}{suffix}"


def _pct_rank(value: float, all_values: list[float], higher_is_better: bool = True) -> str:
    """Return an emoji indicator showing how a value ranks among peers."""
    if len(all_values) < 2:
        return ""
    sorted_vals = sorted(all_values, reverse=higher_is_better)
    rank = sorted_vals.index(value)
    if rank == 0:
        return " 🥇"
    if rank == 1:
        return " 🥈"
    if rank == len(all_values) - 1:
        return " 🐢" if higher_is_better else " ⚡"
    return ""


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    """Remove ``<think>…</think>`` blocks produced by chain-of-thought models.

    Some models (e.g. OpenThinker, DeepSeek-R1) wrap their internal reasoning
    in ``<think>`` tags.  These tokens inflate the total count and duration
    without adding visible value to the user.  Stripping them gives a fairer
    throughput figure for models that expose their scratchpad.
    """
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from visible text (words × 1.35, BPE approximation)."""
    return max(1, round(len(text.split()) * 1.35))


def _load_answer_stats(models: list[str]) -> dict[str, dict[str, dict]]:
    """Load token/timing stats from answer files.

    For every answer file two throughput figures are computed:

    * ``tok_per_s`` — raw (includes ``<think>`` tokens from CoT models).
    * ``visible_tok_per_s`` — after stripping ``<think>…</think>`` blocks.

    The visible figure is used for charts and leaderboards; the raw figure
    is shown in the per-model appendix for full transparency.

    Returns
    -------
    dict[model][prompt_id] = {
        "tokens": int,           # total tokens (raw, from Ollama)
        "visible_tokens": int,   # tokens after think-tag removal (estimated)
        "think_tokens": int,     # difference (CoT scratchpad size)
        "duration_s": float,
        "tok_per_s": float,          # raw throughput
        "visible_tok_per_s": float,  # fair throughput
        "category": str,
        "label": str,
    }
    """
    stats: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in (Path(__file__).parent / "results" / "answers").glob("*.md"):
        meta, body = prog._read_frontmatter(f)
        if meta.get("status") != "done":
            continue
        model = meta.get("model", "")
        pid = meta.get("prompt_id", "")
        tokens = float(meta.get("tokens") or 0)
        duration = float(meta.get("duration_s") or 0)

        # Visible-only token estimate after stripping CoT scratchpad
        visible_body = _strip_think_tags(body)
        visible_tokens = _estimate_tokens(visible_body) if visible_body else int(tokens)
        think_tokens = max(0, int(tokens) - visible_tokens)

        tok_per_s = round(tokens / duration, 2) if duration > 0 else 0.0
        visible_tok_per_s = round(visible_tokens / duration, 2) if duration > 0 else 0.0

        stats[model][pid] = {
            "tokens": int(tokens),
            "visible_tokens": visible_tokens,
            "think_tokens": think_tokens,
            "duration_s": duration,
            "tok_per_s": tok_per_s,
            "visible_tok_per_s": visible_tok_per_s,
            "category": meta.get("category", ""),
            "label": meta.get("label", ""),
        }
    return stats


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_report(models: list[str]) -> None:
    """Collect all benchmark data and write a comprehensive ``results/report.md``."""
    console.print("\n[bold cyan]━━━ Phase 4: Generating Report ━━━[/bold cyan]\n")

    # ── Load scores ──
    all_scores = prog.get_all_scores()
    anon_scores: dict[str, dict[str, int]] = defaultdict(dict)
    named_scores: dict[str, dict[str, int]] = defaultdict(dict)
    mech_scores: dict[str, dict[str, int]] = defaultdict(dict)  # mechanical validation
    for s in all_scores:
        mode = s.get("mode", "")
        scores_map = s.get("scores", {})
        pid = s.get("prompt_id", "")
        if mode == "anonymous":
            for model, score in scores_map.items():
                anon_scores[model][pid] = score
        elif mode == "named":
            for model, score in scores_map.items():
                named_scores[model][pid] = score
        elif mode == "mechanical":
            for model, score in scores_map.items():
                mech_scores[model][pid] = score

    # ── Load answer performance stats ──
    perf: dict[str, dict[str, dict]] = _load_answer_stats(models)

    # ── Aggregate performance per model ──
    model_perf: dict[str, dict] = {}
    for model in models:
        pdata = perf.get(model, {})
        durations = [v["duration_s"] for v in pdata.values() if v["duration_s"] > 0]
        tokens_list = [v["tokens"] for v in pdata.values()]
        visible_tokens_list = [v["visible_tokens"] for v in pdata.values()]
        think_tokens_list = [v["think_tokens"] for v in pdata.values()]
        tps_list = [v["tok_per_s"] for v in pdata.values() if v["tok_per_s"] > 0]
        visible_tps_list = [v["visible_tok_per_s"] for v in pdata.values() if v["visible_tok_per_s"] > 0]
        model_perf[model] = {
            "total_tokens": sum(tokens_list),
            "total_visible_tokens": sum(visible_tokens_list),
            "total_think_tokens": sum(think_tokens_list),
            "total_duration_s": sum(durations),
            "avg_duration_s": statistics.mean(durations) if durations else 0,
            "median_duration_s": statistics.median(durations) if durations else 0,
            "min_duration_s": min(durations) if durations else 0,
            "max_duration_s": max(durations) if durations else 0,
            # Raw throughput (includes CoT think tokens)
            "avg_tok_per_s": statistics.mean(tps_list) if tps_list else 0,
            # Visible throughput (think tags stripped — used for fair rankings)
            "avg_visible_tok_per_s": statistics.mean(visible_tps_list) if visible_tps_list else 0,
            "median_tok_per_s": statistics.median(tps_list) if tps_list else 0,
            "prompts_answered": len(pdata),
        }

    # ── Build category → prompt_ids mapping ──
    cat_prompt_map: dict[str, list[str]] = defaultdict(list)
    for cat in CATEGORIES:
        for p in cat.prompts:
            cat_prompt_map[cat.id].append(p.id)

    # ── Leaderboard composite stats ──
    leaderboard = []
    for model in models:
        anon_vals = list(anon_scores[model].values())
        named_vals = list(named_scores[model].values())
        avg_anon = sum(anon_vals) / len(anon_vals) if anon_vals else 0.0
        avg_named = sum(named_vals) / len(named_vals) if named_vals else 0.0
        bias = avg_named - avg_anon
        avg_tps = model_perf[model]["avg_visible_tok_per_s"]  # visible only (think tags stripped)
        # Efficiency = quality × throughput (higher = better quality per unit time)
        efficiency = round(avg_anon * avg_tps, 2) if avg_tps else 0.0
        leaderboard.append({
            "model": model,
            "avg_anon": avg_anon,
            "avg_named": avg_named,
            "bias": bias,
            "avg_tps": avg_tps,
            "efficiency": efficiency,
        })
    leaderboard.sort(key=lambda x: x["avg_anon"], reverse=True)

    # Pre-compute these once — used by both Charts and later sections
    eff_sorted = sorted(leaderboard, key=lambda e: e["efficiency"], reverse=True)
    tps_order = sorted(leaderboard, key=lambda e: e["avg_tps"], reverse=True)

    user_picks = prog.load_user_picks()

    # ── Convenience refs for rank indicators ──
    all_anon = [e["avg_anon"] for e in leaderboard]
    all_tps = [e["avg_tps"] for e in leaderboard]
    all_eff = [e["efficiency"] for e in leaderboard]

    lines: list[str] = []

    # ════════════════════════════════════════════════════════════════════════════
    # HEADER
    # ════════════════════════════════════════════════════════════════════════════
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_prompts_answered = sum(model_perf[m]["prompts_answered"] for m in models)
    total_tokens_generated = sum(model_perf[m]["total_tokens"] for m in models)
    total_time_s = sum(model_perf[m]["total_duration_s"] for m in models)

    lines += [
        "# 🧪 LLM Benchmark Report\n\n",
        f"> **Generated:** {now}  \n",
        f"> **Models benchmarked:** {', '.join(f'`{m}`' for m in models)}  \n",
        f"> **Benchmark prompts:** {len(ALL_PROMPTS)} across {len(CATEGORIES)} categories  \n",
        f"> **Total answers collected:** {total_prompts_answered}  \n",
        f"> **Total tokens generated:** {total_tokens_generated:,}  \n",
        f"> **Total inference time:** {total_time_s / 60:.1f} min  \n\n",
        "---\n\n",
        "## Quick Summary\n\n",
        "> Use this table to pick the right model for your use-case at a glance.\n\n",
        "| Criterion | Best model |\n",
        "|-----------|------------|\n",
    ]

    best_quality = leaderboard[0]["model"] if leaderboard else "—"
    best_speed_model = max(models, key=lambda m: model_perf[m]["avg_visible_tok_per_s"]) if models else "—"
    best_eff_model = max(leaderboard, key=lambda e: e["efficiency"])["model"] if leaderboard else "—"
    best_value_model = min(models, key=lambda m: model_perf[m]["total_tokens"]) if models else "—"

    lines += [
        f"| 🏅 Best raw quality (anonymous score) | `{best_quality}` |\n",
        f"| ⚡ Fastest (tok/s) | `{best_speed_model}` |\n",
        f"| 🎯 Best quality-per-second (efficiency) | `{best_eff_model}` |\n",
        f"| 💾 Lowest token cost (fewest tokens generated) | `{best_value_model}` |\n",
        "\n---\n\n",
    ]

    # ════════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ════════════════════════════════════════════════════════════════════════════
    lines += [
        "## 📈 Charts\n\n",
        "> Charts render in GitHub, GitLab, VS Code Preview, Obsidian, and most modern Markdown viewers.\n\n",
    ]

    # Chart 1 — Quality scores
    lines += [
        "### Quality Score (Anonymous) — higher is better\n\n",
        _mermaid_bar(
            "Average Anonymous Score (1-10)",
            [e["model"] for e in leaderboard],
            [e["avg_anon"] for e in leaderboard],
            y_label="Score",
            y_max=10,
        ),
        "\n",
    ]

    # Chart 2 — Throughput
    tps_order = sorted(leaderboard, key=lambda e: e["avg_tps"], reverse=True)
    lines += [
        "### Throughput (tok/s) — higher is faster\n\n",
        _mermaid_bar(
            "Average Tokens per Second",
            [e["model"] for e in tps_order],
            [round(e["avg_tps"], 1) for e in tps_order],
            y_label="tok/s",
        ),
        "\n",
    ]

    # Chart 3 — Efficiency
    lines += [
        "### Efficiency (score × tok/s) — quality per unit of time\n\n",
        _mermaid_bar(
            "Efficiency = Avg Score × Avg tok/s",
            [e["model"] for e in eff_sorted],
            [round(e["efficiency"], 1) for e in eff_sorted],
            y_label="Efficiency",
        ),
        "\n",
    ]

    # Chart 4 — Per-category scores as grouped ASCII bars (Mermaid doesn't support grouped bars well)
    lines += [
        "### Per-Category Score Overview (Anonymous) — ASCII bar chart\n\n",
        "> Format: `████░░░░░░` score (10-point scale)  \n\n",
    ]
    for cat in CATEGORIES:
        lines.append(f"**{cat.name}**  \n")
        for model in models:
            cat_s = [anon_scores[model].get(pid) for pid in cat_prompt_map[cat.id] if anon_scores[model].get(pid) is not None]
            avg = sum(cat_s) / len(cat_s) if cat_s else 0.0
            lines.append(f"- `{model}`: {_ascii_bar(model, avg, 10)}  \n")
        lines.append("\n")

    lines.append("---\n\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — QUALITY LEADERBOARD
    # ════════════════════════════════════════════════════════════════════════════
    lines += [
        "## 🏆 Section 1 — Quality Leaderboard\n\n",
        "> Scores are **1–10** averaged across all prompts in each pass.  \n",
        "> **Bias** = Named score − Anonymous score.  ",
        "Positive bias means the model gives higher scores when it can see model names.\n\n",
        "| Rank | Model | Avg Anon | Avg Named | Bias | Efficiency (score × tok/s) |\n",
        "|------|-------|----------|-----------|------|----------------------------|\n",
    ]

    for rank, entry in enumerate(leaderboard, 1):
        m = entry["model"]
        bias_str = f"+{entry['bias']:.2f}" if entry["bias"] >= 0 else f"{entry['bias']:.2f}"
        anon_flag = _pct_rank(entry["avg_anon"], all_anon)
        eff_flag = _pct_rank(entry["efficiency"], all_eff)
        lines.append(
            f"| {rank} | `{m}` "
            f"| {entry['avg_anon']:.2f}{anon_flag} "
            f"| {entry['avg_named']:.2f} "
            f"| {bias_str} "
            f"| {entry['efficiency']:.2f}{eff_flag} |\n"
        )

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — PER-CATEGORY QUALITY
    # ════════════════════════════════════════════════════════════════════════════
    col_header = "| Category | " + " | ".join(f"`{m}`" for m in models) + " |\n"
    col_sep = "|----------|" + "|".join(["---:"] * len(models)) + "|\n"

    lines += [
        "---\n\n",
        "## 📊 Section 2 — Per-Category Quality Scores\n\n",
        "### 2a — Anonymous Judging\n\n",
        "> Answers labelled A/B/C — no model names shown to the judge.\n\n",
        col_header, col_sep,
    ]

    for cat in CATEGORIES:
        # Compute avg per model for this category to determine rank
        cat_avgs: list[Optional[float]] = []
        for m in models:
            cs = [anon_scores[m].get(pid) for pid in cat_prompt_map[cat.id] if anon_scores[m].get(pid) is not None]
            cat_avgs.append(sum(cs) / len(cs) if cs else None)
        non_none_avgs = [v for v in cat_avgs if v is not None]

        row_vals = []
        for model, avg in zip(models, cat_avgs):
            if avg is None:
                row_vals.append("—")
            else:
                flag = _pct_rank(avg, non_none_avgs)
                row_vals.append(_fmt(avg) + flag)
        lines.append(f"| {cat.name} | " + " | ".join(row_vals) + " |\n")

    lines += [
        "\n### 2b — Named Judging\n\n",
        "> Answers shown with model names — captures self-preference bias.\n\n",
        col_header, col_sep,
    ]

    for cat in CATEGORIES:
        row_vals = []
        for model in models:
            cat_s = [named_scores[model].get(pid) for pid in cat_prompt_map[cat.id]]
            cat_s = [s for s in cat_s if s is not None]
            avg = sum(cat_s) / len(cat_s) if cat_s else None
            row_vals.append(_fmt(avg) if avg is not None else "—")
        lines.append(f"| {cat.name} | " + " | ".join(row_vals) + " |\n")

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PERFORMANCE STATISTICS
    # ════════════════════════════════════════════════════════════════════════════
    lines += [
        "---\n\n",
        "## ⚡ Section 3 — Performance & Latency\n\n",
        "> All times are **wall-clock seconds** measured on an M1 MacBook Pro 16 GB  \n",
        "> running Ollama locally.  tok/s = tokens generated ÷ wall-clock time.\n\n",
    ]

    # Section 3a — use visible tok/s for ranking, show raw in parens
    tps_all = [model_perf[m]["avg_visible_tok_per_s"] for m in models]
    dur_all = [model_perf[m]["avg_duration_s"] for m in models]

    lines += [
        "### 3a — Throughput & Token Summary\n\n",
        "> tok/s is computed on **visible tokens only** (<think>…</think> scratchpad stripped)  \n",
        "> to give a fair comparison across standard and chain-of-thought models.\n\n",
        "| Model | Avg resp (s) | Median (s) | Min (s) | Max (s) | Visible tok/s | Raw tok/s | Visible tokens | Think tokens | Total tokens |\n",
        "|-------|-------------|------------|---------|---------|--------------|-----------|---------------|-------------|-------------|\n",
    ]

    for model in models:
        p = model_perf[model]
        vis_tps = p["avg_visible_tok_per_s"]
        raw_tps = p["avg_tok_per_s"]
        tps_flag = _pct_rank(vis_tps, tps_all)
        dur_flag = _pct_rank(p["avg_duration_s"], dur_all, higher_is_better=False)
        lines.append(
            f"| `{model}` "
            f"| {_fmt(p['avg_duration_s'])}{dur_flag} "
            f"| {_fmt(p['median_duration_s'])} "
            f"| {_fmt(p['min_duration_s'])} "
            f"| {_fmt(p['max_duration_s'])} "
            f"| {_fmt(vis_tps)}{tps_flag} "
            f"| {_fmt(raw_tps)} "
            f"| {p['total_visible_tokens']:,} "
            f"| {p['total_think_tokens']:,} "
            f"| {p['total_tokens']:,} |\n"
        )

    lines.append("\n")

    # 3b — Per-category latency heatmap
    lines += [
        "### 3b — Average Response Time by Category (seconds)\n\n",
        "> Lower is better. Helps identify which task types are slow on which models.\n\n",
        col_header, col_sep,
    ]

    for cat in CATEGORIES:
        row_vals = []
        all_cat_durs = []
        for model in models:
            cat_durs = [
                perf[model][pid]["duration_s"]
                for pid in cat_prompt_map[cat.id]
                if pid in perf.get(model, {})
            ]
            avg_dur = statistics.mean(cat_durs) if cat_durs else None
            if avg_dur is not None:
                all_cat_durs.append(avg_dur)
            row_vals.append((_fmt(avg_dur, 1, "s"), avg_dur))

        # Add rank indicators per row
        formatted = []
        for val_str, val in row_vals:
            if val is not None and len(all_cat_durs) > 1:
                flag = _pct_rank(val, all_cat_durs, higher_is_better=False)
                formatted.append(val_str + flag)
            else:
                formatted.append(val_str)
        lines.append(f"| {cat.name} | " + " | ".join(formatted) + " |\n")

    lines.append("\n")

    # 3c — Per-category throughput (tok/s)
    lines += [
        "### 3c — Average Throughput by Category (tok/s)\n\n",
        "> Higher is better.\n\n",
        col_header, col_sep,
    ]

    for cat in CATEGORIES:
        row_vals_raw = []
        for model in models:
            cat_tps = [
                perf[model][pid]["visible_tok_per_s"]
                for pid in cat_prompt_map[cat.id]
                if pid in perf.get(model, {}) and perf[model][pid]["visible_tok_per_s"] > 0
            ]
            avg_tps = statistics.mean(cat_tps) if cat_tps else None
            row_vals_raw.append(avg_tps)

        non_none = [v for v in row_vals_raw if v is not None]
        formatted = []
        for val in row_vals_raw:
            if val is not None:
                flag = _pct_rank(val, non_none)
                formatted.append(_fmt(val, 1) + flag)
            else:
                formatted.append("—")
        lines.append(f"| {cat.name} | " + " | ".join(formatted) + " |\n")

    lines.append("\n")

    # 3d — Per-category token verbosity
    lines += [
        "### 3d — Average Tokens Generated by Category\n\n",
        "> Reflects answer verbosity. High token counts increase latency and energy use.\n\n",
        col_header, col_sep,
    ]

    for cat in CATEGORIES:
        row_vals_raw = []
        for model in models:
            cat_tok = [
                perf[model][pid]["tokens"]
                for pid in cat_prompt_map[cat.id]
                if pid in perf.get(model, {})
            ]
            avg_tok = statistics.mean(cat_tok) if cat_tok else None
            row_vals_raw.append(avg_tok)

        non_none = [v for v in row_vals_raw if v is not None]
        formatted = []
        for val in row_vals_raw:
            if val is not None:
                # Lower token count = more concise, neither strictly better nor worse —
                # just show the value.
                formatted.append(f"{int(val):,}")
            else:
                formatted.append("—")
        lines.append(f"| {cat.name} | " + " | ".join(formatted) + " |\n")

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — EFFICIENCY FRONTIER
    # ════════════════════════════════════════════════════════════════════════════
    lines += [
        "---\n\n",
        "## 🎯 Section 4 — Quality-Efficiency Frontier\n\n",
        "> **Efficiency = Avg anonymous score × Avg tok/s**  \n",
        "> This combines quality and speed into a single number.  \n",
        "> The higher the score, the more quality you get per unit of time.  \n",
        "> Use this to pick the model that gives the best return on compute.\n\n",
        "| Model | Avg Score | Avg tok/s | Efficiency | Verdict |\n",
        "|-------|-----------|-----------|------------|--------|\n",
    ]

    best_eff = eff_sorted[0]["efficiency"] if eff_sorted else 1

    for entry in eff_sorted:
        m = entry["model"]
        rel = entry["efficiency"] / best_eff if best_eff else 0
        bar = "█" * int(rel * 10) + "░" * (10 - int(rel * 10))
        if rel >= 0.9:
            verdict = "✅ Best overall"
        elif entry["avg_anon"] == max(e["avg_anon"] for e in leaderboard):
            verdict = "🧠 Highest quality"
        elif entry["avg_tps"] == max(e["avg_tps"] for e in leaderboard):
            verdict = "⚡ Fastest"
        else:
            verdict = "—"
        lines.append(
            f"| `{m}` | {entry['avg_anon']:.2f} | {entry['avg_tps']:.1f} "
            f"| {entry['efficiency']:.2f} `{bar}` | {verdict} |\n"
        )

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 5 — BIAS ANALYSIS
    # ════════════════════════════════════════════════════════════════════════════
    lines += [
        "---\n\n",
        "## 🔍 Section 5 — Self-Bias Analysis\n\n",
        "> When a model scores answers **knowing which model wrote them**, its scores\n"
        "> may differ from when it scores blindly.  \n",
        "> **Positive bias** = model inflates scores when names are visible (self-preferring).  \n",
        "> **Negative bias** = model deflates scores with names visible (self-critical).  \n\n",
        "| Model | Avg Anon | Avg Named | Bias | Interpretation |\n",
        "|-------|----------|-----------|------|----------------|\n",
    ]

    for entry in leaderboard:
        bias = entry["bias"]
        if abs(bias) < 0.3:
            interp = "Consistent — scores stable with/without names"
        elif bias > 0:
            interp = f"⚠️ Self-favoring — inflates by {bias:.2f} pts when names shown"
        else:
            interp = f"Self-critical — deflates by {abs(bias):.2f} pts when names shown"
        bias_str = f"+{bias:.2f}" if bias >= 0 else f"{bias:.2f}"
        lines.append(
            f"| `{entry['model']}` | {entry['avg_anon']:.2f} | {entry['avg_named']:.2f} "
            f"| {bias_str} | {interp} |\n"
        )

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 6 — USER PICKS
    # ════════════════════════════════════════════════════════════════════════════
    lines += [
        "---\n\n",
        "## 👤 Section 6 — Your Manual Picks\n\n",
        "> These are prompts where *you* chose the best answer after reading them side-by-side.\n",
        "> This is the most direct signal of real-world usefulness.\n\n",
    ]

    if user_picks:
        pick_wins: dict[str, int] = defaultdict(int)
        lines += [
            "| Prompt ID | Category | Label | Your Winner |\n",
            "|-----------|----------|-------|-------------|\n",
        ]
        for pick in user_picks:
            pid = pick["prompt_id"]
            p_obj = ALL_PROMPTS.get(pid)
            label = p_obj.label if p_obj else pid
            lines.append(
                f"| `{pid}` | {pick['category']} | {label} | `{pick['winner']}` |\n"
            )
            pick_wins[pick["winner"]] += 1

        lines.append("\n**Pick tally:**\n\n")
        total_picks = sum(pick_wins.values())
        for model_name, count in sorted(pick_wins.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / total_picks if total_picks else 0
            bar = "█" * count + "░" * (total_picks - count)
            lines.append(f"- `{model_name}`: {count}/{total_picks} ({pct:.0f}%) `{bar}`\n")
    else:
        lines.append("_No manual picks recorded._\n")

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 7 — DECISION GUIDE
    # ════════════════════════════════════════════════════════════════════════════

    lines += [
        "---\n\n",
        "## 🗺️ Section 7 — Decision Guide\n\n",
        "Use the table below to match your primary use-case to the recommended model.\n\n",
        "| Use-case | Recommended model | Why |\n",
        "|----------|------------------|-----|\n",
    ]

    # Best quality
    bq = leaderboard[0]["model"] if leaderboard else "—"
    lines.append(f"| Best overall quality | `{bq}` | Highest avg anonymous score |\n")

    # Fastest
    bf = max(models, key=lambda m: model_perf[m]["avg_visible_tok_per_s"]) if models else "—"
    bf_tps = model_perf[bf]["avg_visible_tok_per_s"] if models else 0
    lines.append(
        f"| Interactive / low-latency use | `{bf}` | {bf_tps:.1f} tok/s avg — fastest response |\n"
    )

    # Best efficiency
    be = eff_sorted[0]["model"] if eff_sorted else "—"
    lines.append(
        f"| Best quality per unit time | `{be}` | Efficiency score {eff_sorted[0]['efficiency']:.2f} |\n"
    )

    # Most concise
    mc = min(models, key=lambda m: model_perf[m]["total_tokens"]) if models else "—"
    mc_tok = model_perf[mc]["total_tokens"] if models else 0
    lines.append(
        f"| Concise / low-resource | `{mc}` | {mc_tok:,} total tokens — most concise output |\n"
    )

    # Human favourite
    if user_picks:
        fav = max(pick_wins, key=pick_wins.get) if pick_wins else "—"
        fav_wins = pick_wins.get(fav, 0)
        lines.append(
            f"| Human preference | `{fav}` | You picked it {fav_wins}× in manual review |\n"
        )

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 7b — FORMAT COMPLIANCE (mechanical validation)
    # ════════════════════════════════════════════════════════════════════════════
    fmt_prompt_ids = [pid for pid in ALL_PROMPTS if pid in FORMAT_CHECKED_PROMPT_IDS]
    if fmt_prompt_ids and any(mech_scores[m] for m in models):
        lines += [
            "---\n\n",
            "## 🤖 Section 7b — Mechanical Format Compliance\n\n",
            "> These prompts have **objectively verifiable** output formats (JSON schema, YAML frontmatter, enum values).  \n",
            "> Scores are computed by a deterministic checker \u2014 **no LLM involved**.  \n",
            "> 10 = all schema fields correct and parseable · 0 = unparseable or wrong format.\n\n",
        ]

        # Summary chart
        mech_avgs = []
        for model in models:
            vals = [mech_scores[model].get(pid, 0) for pid in fmt_prompt_ids]
            mech_avgs.append(sum(vals) / len(vals) if vals else 0.0)

        lines += [
            _mermaid_bar(
                "Mechanical Format Compliance Score (0-10)",
                models,
                mech_avgs,
                y_label="Compliance Score",
                y_max=10,
            ),
            "\n",
        ]

        # Detailed table
        fmt_cat_ids = ["structured_output", "data_extraction"]
        fmt_per_cat: dict[str, list[str]] = {}
        for cat in CATEGORIES:
            if cat.id in fmt_cat_ids:
                fmt_per_cat[cat.name] = [p.id for p in cat.prompts]

        lines += [
            "| Prompt | Category | " + " | ".join(f"`{m}`" for m in models) + " |\n",
            "|--------|----------|" + "|".join(["---:"] * len(models)) + "|\n",
        ]

        for cat_name, pids in fmt_per_cat.items():
            for pid in pids:
                p_obj = ALL_PROMPTS.get(pid)
                label = p_obj.label if p_obj else pid
                row = []
                for model in models:
                    score = mech_scores[model].get(pid)
                    if score is None:
                        row.append("—")
                    elif score >= 8:
                        row.append(f"✅ {score}")
                    elif score >= 5:
                        row.append(f"⚠️ {score}")
                    else:
                        row.append(f"❌ {score}")
                lines.append(f"| {label} | {cat_name} | " + " | ".join(row) + " |\n")

        # Per-model avg
        lines.append("\n**Average mechanical compliance scores:**\n\n")
        avg_list = sorted(zip(models, mech_avgs), key=lambda x: x[1], reverse=True)
        for model_name, avg in avg_list:
            flag = _pct_rank(avg, [a for _, a in avg_list])
            lines.append(f"- `{model_name}`: **{avg:.1f}/10**{flag}\n")

        lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 7c — SCORING RELIABILITY
    # ════════════════════════════════════════════════════════════════════════════
    import json as _json

    stats_path = Path(__file__).parent / "results" / "scoring_stats.json"
    scoring_stats: dict = {}
    if stats_path.exists():
        try:
            scoring_stats = _json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Build coverage map: for each prompt_id, which models have an anon score?
    # anon_scores[model][pid] is already loaded above.
    prompt_coverage: dict[str, list[str]] = {}  # pid -> list of models WITH a score
    for pid in ALL_PROMPTS:
        prompt_coverage[pid] = [m for m in models if anon_scores[m].get(pid) is not None]

    incomplete_prompts = {pid: ms for pid, ms in prompt_coverage.items() if len(ms) < len(models) and ms}

    lines += [
        "---\n\n",
        "## 📊 Section 7c — Scoring Reliability\n\n",
        "> How well did each model follow the JSON scoring format?  \n",
        "> A **failed parse** means a scoring call returned output that couldn't be mapped to scores — those prompts are missing from leaderboards.\n\n",
    ]

    # ── 7c-i: Success rate chart (only if stats file was written this run) ──
    if scoring_stats:
        rate_vals = [round(scoring_stats.get(m, {}).get("success_rate", 1.0) * 100, 1) for m in models]
        lines += [
            "### Parse Success Rate (%)\n\n",
            _mermaid_bar(
                "Scoring JSON Parse Success Rate (%)",
                models,
                rate_vals,
                y_label="%",
                y_max=100,
            ),
            "\n",
            "| Model | Anon OK | Named OK | Anon Fail | Named Fail | Success Rate |\n",
            "|-------|---------|----------|-----------|------------|-------------|\n",
        ]
        for model in models:
            st = scoring_stats.get(model, {})
            rate = st.get("success_rate", 1.0)
            rate_flag = " 🥇" if rate == 1.0 else (" ⚠️" if rate < 0.9 else "")
            a_fail = st.get("anon_fail", [])
            n_fail = st.get("named_fail", [])
            lines.append(
                f"| `{model}` "
                f"| {st.get('anon_ok', '—')} "
                f"| {st.get('named_ok', '—')} "
                f"| {len(a_fail)} "
                f"| {len(n_fail)} "
                f"| **{rate * 100:.1f}%**{rate_flag} |\n"
            )
        lines.append("\n")

        # List failed prompts per model
        any_fail = any(
            scoring_stats.get(m, {}).get("anon_fail") or scoring_stats.get(m, {}).get("named_fail")
            for m in models
        )
        if any_fail:
            lines.append("**Failed prompts by model:**\n\n")
            for model in models:
                st = scoring_stats.get(model, {})
                all_fail = sorted(set(st.get("anon_fail", []) + st.get("named_fail", [])))
                if all_fail:
                    lines.append(f"- `{model}`: {', '.join(f'`{p}`' for p in all_fail)}\n")
            lines.append("\n")
    else:
        lines.append("> *Run the benchmark (scoring phase) to populate parse success stats.*\n\n")

    # ── 7c-ii: Prompt coverage table — ⚠ where not all models scored ──
    lines += [
        "### Score Coverage per Prompt\n\n",
        "> ⚠️ = fewer than all models produced a valid score for this prompt.\n\n",
        "| Category | Prompt | Models scored | Coverage |\n",
        "|----------|--------|--------------|----------|\n",
    ]
    for cat in CATEGORIES:
        for p_obj in cat.prompts:
            pid = p_obj.id
            scored = prompt_coverage.get(pid, [])
            n = len(scored)
            total_m = len(models)
            if n == total_m:
                cov = f"✅ {n}/{total_m}"
            elif n == 0:
                cov = f"❌ 0/{total_m} — no data"
            else:
                missing = [m for m in models if m not in scored]
                cov = f"⚠️ {n}/{total_m} — missing: {', '.join(f'`{m}`' for m in missing)}"
            lines.append(f"| {cat.name} | {p_obj.label} | {n} | {cov} |\n")

    lines.append("\n")

    # ════════════════════════════════════════════════════════════════════════════
    # SECTION 8 — PER-MODEL APPENDIX
    # ════════════════════════════════════════════════════════════════════════════
    lines += [
        "---\n\n",
        "## 📎 Section 8 — Per-Model Detail\n\n",
    ]

    for model in models:
        p = model_perf[model]
        lines.append(f"### `{model}`\n\n")
        lines += [
            f"- **Total prompts answered:** {p['prompts_answered']}  \n",
            f"- **Total tokens generated:** {p['total_tokens']:,}  \n",
            f"- **Total inference time:** {p['total_duration_s']:.1f} s ({p['total_duration_s']/60:.1f} min)  \n",
            f"- **Avg response time:** {p['avg_duration_s']:.2f} s  \n",
            f"- **Avg throughput (visible):** {p['avg_visible_tok_per_s']:.2f} tok/s  \n",
            f"- **Avg throughput (raw incl. think):** {p['avg_tok_per_s']:.2f} tok/s  \n\n",
        ]

        lines.append("| Category | Prompt | Anon Score | Named Score | Duration (s) | Tokens | tok/s |\n")
        lines.append("|----------|--------|-----------|------------|-------------|--------|-------|\n")
        for cat in CATEGORIES:
            for p_obj in cat.prompts:
                pid = p_obj.id
                anon = anon_scores[model].get(pid)
                named = named_scores[model].get(pid)
                pstat = perf.get(model, {}).get(pid, {})
                dur = pstat.get("duration_s")
                tok = pstat.get("tokens")
                tps = pstat.get("visible_tok_per_s")
                lines.append(
                    f"| {cat.name} | {p_obj.label} "
                    f"| {anon if anon is not None else '—'} "
                    f"| {named if named is not None else '—'} "
                    f"| {_fmt(dur, 1)} "
                    f"| {f'{tok:,}' if tok else '—'} "
                    f"| {_fmt(tps, 1)} |\n"
                )
        lines.append("\n")

    # ── Write file ──
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_text = "".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    console.print(f"[bold green]✓ Report written to:[/bold green] {REPORT_PATH}\n")

    if EXPORT_PDF:
        if MarkdownPdf is None:
            console.print("[bold yellow]⚠ EXPORT_PDF is true, but `markdown-pdf` is not installed. Skipping PDF generation.[/bold yellow]")
        else:
            pdf_path = REPORT_PATH.with_suffix(".pdf")
            try:
                console.print(f"  [dim]Generating PDF at {pdf_path}...[/dim]")
                pdf = MarkdownPdf(toc_level=2)
                # markdown_pdf requires adding sections
                pdf.add_section(Section(report_text, toc=False))
                pdf.save(str(pdf_path))
                console.print(f"[bold green]✓ PDF report exported to {pdf_path}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]✗ Failed to generate PDF: {e}[/bold red]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_report(sys.argv[1:])
    else:
        generate_report([])
