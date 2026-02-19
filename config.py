"""
config.py — Benchmark configuration: categories, prompts, and scoring templates.

This is the single place to add, remove, or edit benchmark prompts.
Each ``Category`` holds a list of ``Prompt`` objects.  The flat
``ALL_PROMPTS`` dict is built automatically and used everywhere else in
the codebase to look up a prompt by its unique ``id``.

Tuning constants
----------------
* ``OLLAMA_BASE_URL`` – base URL of the running Ollama server.
* ``USER_PICKS``      – number of prompts shown to the user for manual ranking.
"""

from dataclasses import dataclass, field
from typing import List

OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class Prompt:
    """A single benchmark prompt.

    Attributes
    ----------
    id:       Unique snake_case identifier (e.g. ``math_2``).  Used as the
              filename stem for answer/score files.
    category: Parent category id (e.g. ``math``).
    label:    Human-readable short name shown in the UI.
    system:   System prompt injected into the model before the user turn.
    user:     The actual question / task sent to the model.
    """

    id: str
    category: str
    label: str
    system: str
    user: str


@dataclass
class Category:
    """A benchmark category grouping related prompts.

    Attributes
    ----------
    id:      Snake_case identifier (e.g. ``instruction_following``).
    name:    Emoji-prefixed display name shown in tables and the report.
    prompts: Ordered list of :class:`Prompt` objects.
    """

    id: str
    name: str
    prompts: List[Prompt]


# ---------------------------------------------------------------------------
# Benchmark prompts
# ---------------------------------------------------------------------------

CATEGORIES: List[Category] = [
    Category(
        id="summarization",
        name="📝 Summarization",
        prompts=[
            Prompt(
                id="sum_1",
                category="summarization",
                label="News article summary",
                system="You are a professional editor. Summarize the following text concisely.",
                user=(
                    "Summarize the following article in 3 bullet points, each no longer than 20 words:\n\n"
                    "Artificial intelligence is reshaping nearly every industry. In healthcare, AI systems are now "
                    "diagnosing diseases from medical images with accuracy rivaling that of experienced physicians. "
                    "In finance, machine learning models detect fraudulent transactions in milliseconds. The legal "
                    "sector is seeing AI tools that can review thousands of contracts in hours, a task that would "
                    "take a team of lawyers weeks. Despite the benefits, experts warn of risks: job displacement, "
                    "bias in automated decisions, and the growing opacity of algorithmic systems. Governments "
                    "worldwide are racing to draft regulations that balance innovation with protection. A recent "
                    "EU framework mandates transparency disclosures for high-risk AI, while the US opts for "
                    "voluntary guidelines. The debate over who is responsible when an AI makes a consequential "
                    "error remains unresolved."
                ),
            ),
            Prompt(
                id="sum_2",
                category="summarization",
                label="Technical document summary",
                system="You are a technical writer. Summarize complex technical content clearly.",
                user=(
                    "Read the following technical description and write a one-paragraph executive summary "
                    "suitable for a non-technical audience:\n\n"
                    "A transformer-based large language model is a deep neural network architecture that uses "
                    "self-attention mechanisms to process sequences of tokens in parallel rather than sequentially. "
                    "The model is pre-trained on a massive corpus using a next-token prediction objective, causing "
                    "it to learn statistical patterns across language. Fine-tuning with reinforcement learning "
                    "from human feedback (RLHF) aligns the model's outputs to human preferences. At inference "
                    "time, the model auto-regressively generates tokens by sampling from a probability "
                    "distribution conditioned on all preceding context tokens within its fixed context window."
                ),
            ),
        ],
    ),
    Category(
        id="qa",
        name="❓ Question & Answer",
        prompts=[
            Prompt(
                id="qa_1",
                category="qa",
                label="Factual science Q&A",
                system="You are a knowledgeable assistant. Answer factually and concisely.",
                user=(
                    "Answer the following questions accurately and concisely:\n"
                    "1. What is the speed of light in a vacuum?\n"
                    "2. How many bones are in the adult human body?\n"
                    "3. What element has the atomic number 79?\n"
                    "4. What is the powerhouse of the cell?\n"
                    "5. In what year did the first moon landing occur?"
                ),
            ),
            Prompt(
                id="qa_2",
                category="qa",
                label="Multi-hop reasoning Q&A",
                system="You are an intelligent assistant. Think step by step before answering.",
                user=(
                    "If Alice is taller than Bob, and Bob is taller than Carol, and Carol is taller than Dave, "
                    "who is the shortest? Now, if Eve is shorter than Dave but taller than Frank, list all six "
                    "people from tallest to shortest."
                ),
            ),
        ],
    ),
    Category(
        id="reasoning",
        name="🧠 Reasoning",
        prompts=[
            Prompt(
                id="reason_1",
                category="reasoning",
                label="Logical puzzle",
                system="You are a logical reasoning expert. Show your work.",
                user=(
                    "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All three labels are wrong. "
                    "You may pick one fruit from one box only. Which box do you pick from, and how do you "
                    "correctly label all three boxes? Explain your reasoning step by step."
                ),
            ),
            Prompt(
                id="reason_2",
                category="reasoning",
                label="Causal reasoning",
                system="You are an expert in critical thinking and causal analysis.",
                user=(
                    "A company launches a new product and its stock price rises 20%. The CEO claims the price "
                    "rise was directly caused by the product launch. Identify at least three alternative "
                    "explanations for the stock price rise that the CEO is ignoring, and explain what evidence "
                    "would be needed to confirm or deny each explanation."
                ),
            ),
        ],
    ),
    Category(
        id="math",
        name="🔢 Math",
        prompts=[
            Prompt(
                id="math_1",
                category="math",
                label="Arithmetic & algebra",
                system="You are a precise mathematician. Show all steps.",
                user=(
                    "Solve each of the following:\n"
                    "1. 347 × 58 = ?\n"
                    "2. √(144) + √(225) = ?\n"
                    "3. Solve for x: 3x² - 12x + 9 = 0\n"
                    "4. What is 15% of 840?"
                ),
            ),
            Prompt(
                id="math_2",
                category="math",
                label="Word problem",
                system="You are a patient math tutor. Solve word problems step by step.",
                user=(
                    "A train leaves City A at 9:00 AM traveling at 80 km/h toward City B, which is 300 km away. "
                    "Another train leaves City B at the same time traveling toward City A at 70 km/h. "
                    "At what time will the trains meet, and how far from City A will the meeting point be? "
                    "Show all your work."
                ),
            ),
            Prompt(
                id="math_3",
                category="math",
                label="Probability",
                system="You are a statistician. Explain your reasoning clearly.",
                user=(
                    "A bag contains 5 red balls, 3 blue balls, and 2 green balls.\n"
                    "1. What is the probability of drawing a red ball on the first draw?\n"
                    "2. If a red ball is drawn and NOT replaced, what is the probability the second draw is also red?\n"
                    "3. What is the probability of drawing two red balls in a row (without replacement)?\n"
                    "Express all answers as fractions and percentages."
                ),
            ),
        ],
    ),
    Category(
        id="code",
        name="💻 Code",
        prompts=[
            Prompt(
                id="code_1",
                category="code",
                label="Write a function",
                system="You are an expert software engineer. Write clean, idiomatic code with comments.",
                user=(
                    "Write a Python function called `find_duplicates` that takes a list of integers and returns "
                    "a sorted list of all integers that appear more than once. The function should run in O(n) "
                    "time complexity. Include type hints, a docstring, and at least 3 usage examples with expected output."
                ),
            ),
            Prompt(
                id="code_2",
                category="code",
                label="Debug a snippet",
                system="You are a senior code reviewer. Find all bugs and explain how to fix them.",
                user=(
                    "The following Python function is supposed to compute the Fibonacci sequence up to n terms "
                    "and return the result as a list. Find all bugs, explain each one, and provide the corrected code:\n\n"
                    "```python\n"
                    "def fibonacci(n):\n"
                    "    if n = 0:\n"
                    "        return []\n"
                    "    elif n == 1\n"
                    "        return [0]\n"
                    "    sequence = [0, 1]\n"
                    "    for i in range(2, n):\n"
                    "        next_val = sequence[i-1] + sequence[i-2]\n"
                    "        sequence.append(next_val)\n"
                    "    return sequence[n]  # should return full list\n"
                    "```"
                ),
            ),
        ],
    ),
    Category(
        id="instruction_following",
        name="📋 Instruction Following",
        prompts=[
            Prompt(
                id="instr_1",
                category="instruction_following",
                label="Multi-constraint formatting",
                system="You follow instructions precisely. Do not deviate from any constraint.",
                user=(
                    "Write a short bio for a fictional scientist named Dr. Lyra Osei. Follow ALL of these constraints:\n"
                    "- Exactly 4 sentences long\n"
                    "- First sentence must start with 'Dr.'\n"
                    "- Must mention exactly two scientific fields\n"
                    "- Must include a specific year (any year between 1990 and 2020)\n"
                    "- Must not use the word 'renowned' or 'notable'\n"
                    "- Final sentence must be a question"
                ),
            ),
            Prompt(
                id="instr_2",
                category="instruction_following",
                label="Acrostic poem",
                system="You are a creative writer who follows instructions exactly.",
                user=(
                    "Write an acrostic poem using the word PYTHON where:\n"
                    "- Each line must start with the corresponding letter of PYTHON\n"
                    "- Each line must relate to programming or technology\n"
                    "- Each line must be exactly 8 words long\n"
                    "- The poem must rhyme (ABABAB scheme)\n"
                    "After the poem, list how many words are in each line to verify."
                ),
            ),
        ],
    ),
    Category(
        id="agentic",
        name="🤖 Agentic Flow",
        prompts=[
            Prompt(
                id="agent_1",
                category="agentic",
                label="Multi-step task planning",
                system=(
                    "You are an autonomous AI assistant. When given a task, break it into clear, "
                    "executable steps. For each step, specify what tool or action would be needed "
                    "(e.g., web_search, read_file, execute_code, send_email)."
                ),
                user=(
                    "Task: A user asks you to research the top 3 Python web frameworks in 2024, "
                    "compare them in a markdown table, save it to a file called 'frameworks.md', "
                    "and then send that file to 'team@acme.com' with a subject line summarizing the comparison.\n\n"
                    "Create a numbered, step-by-step action plan. For each step include:\n"
                    "- The action name\n"
                    "- The tool you would call\n"
                    "- The exact inputs to that tool\n"
                    "- What you'd do with the output before the next step"
                ),
            ),
            Prompt(
                id="agent_2",
                category="agentic",
                label="Error recovery planning",
                system=(
                    "You are a robust AI agent. You plan for failures and have contingency steps."
                ),
                user=(
                    "You are an AI agent that needs to scrape product prices from a website, "
                    "store them in a database, and send a daily report. Design the full agent loop including:\n"
                    "1. The happy path (all steps succeed)\n"
                    "2. At least 3 failure modes and how the agent should recover from each\n"
                    "3. How the agent should decide when to escalate to a human\n"
                    "Be specific about state management between iterations."
                ),
            ),
        ],
    ),
    Category(
        id="creative",
        name="✍️ Creative Writing",
        prompts=[
            Prompt(
                id="creative_1",
                category="creative",
                label="Short story with constraints",
                system="You are a gifted creative writer. Write vivid, engaging prose.",
                user=(
                    "Write a short story (150–200 words) set in a world where gravity reverses every 24 hours. "
                    "The story must have:\n"
                    "- A named protagonist\n"
                    "- A clear beginning, middle, and end\n"
                    "- At least one moment of tension\n"
                    "- An ending sentence that echoes the opening sentence"
                ),
            ),
        ],
    ),
]

# Flat lookup by prompt ID
ALL_PROMPTS: dict[str, Prompt] = {
    p.id: p for cat in CATEGORIES for p in cat.prompts
}

# ---------------------------------------------------------------------------
# Scoring prompt templates
# ---------------------------------------------------------------------------

ANONYMOUS_SCORING_SYSTEM = (
    "You are an impartial expert evaluator. You will be shown a question and several answers "
    "labelled with letters (A, B, C, …). Evaluate each answer on a scale from 1 to 10 based on "
    "accuracy, completeness, clarity, and usefulness. Do NOT guess who wrote the answers — treat "
    "them purely on merit. Respond ONLY with valid JSON in this exact format:\n"
    '{"scores": {"A": <int>, "B": <int>, ...}, "rationale": {"A": "<string>", "B": "<string>", ...}}'
)

NAMED_SCORING_SYSTEM = (
    "You are an expert evaluator. You will be shown a question and several answers labelled with "
    "model names. Evaluate each answer on a scale from 1 to 10 based on accuracy, completeness, "
    "clarity, and usefulness. Respond ONLY with valid JSON in this exact format:\n"
    '{"scores": {"<model_name>": <int>, ...}, "rationale": {"<model_name>": "<string>", ...}}'
)

USER_PICKS = 5  # how many prompts to show the user for manual ranking
