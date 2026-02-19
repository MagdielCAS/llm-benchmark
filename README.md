# LLM Benchmark — Local Ollama Models

A CLI script that benchmarks your local Ollama models across 8 categories and produces a comprehensive markdown report.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Run once:

```bash
cd llm-benchmark
uv sync
```

## Usage

```bash
uv run python benchmark.py
```

The script will:
1. **Detect** all installed Ollama models and ask which to benchmark
2. **Ask** whether to resume a previous run or start fresh
3. **Phase 1 — Answers**: Each model answers all prompts across 8 categories
4. **Phase 2 — Scoring**: Each model judges all answers anonymously, then again with model names revealed
5. **Phase 3 — Your picks**: You manually select the best answer for ~5 prompts
6. **Phase 4 — Report**: `results/report.md` is generated with leaderboard, per-category tables, bias analysis, and your picks

> **Resumable**: If the script is interrupted at any point, re-run it and choose **Resume**. All completed answers and scores are saved as markdown files in `results/`.

## Benchmark Categories

| Category | Prompts |
|---|---|
| 📝 Summarization | 2 |
| ❓ Q&A | 2 |
| 🧠 Reasoning | 2 |
| 🔢 Math | 3 |
| 💻 Code | 2 |
| 📋 Instruction Following | 2 |
| 🤖 Agentic Flow | 2 |
| ✍️ Creative Writing | 1 |

## Output Structure

```
results/
├── answers/          # <model>__<prompt_id>.md (one per model × prompt)
├── scores/           # <scoring_model>__<prompt_id>__<mode>.md
├── user_picks.md     # Your manual selections
└── report.md         # Final benchmark report
```

Each file uses YAML frontmatter for structured metadata + full text body.

## Requirements

- [Ollama](https://ollama.ai) running locally (`ollama serve`)
- Python 3.11+, uv
- At least 2 local models pulled (e.g. `ollama pull llama3.2`, `ollama pull mistral`)
