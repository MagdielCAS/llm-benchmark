"""
openrouter_client.py — Thin wrapper around the OpenRouter REST API.

Used exclusively for the external scoring feature.
"""

import time
import httpx
from typing import Optional
from config import OPENROUTER_API_KEY


def generate(
    model: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    timeout: float = 300.0,
) -> tuple[str, int, float]:
    """Call OpenRouter chat completions API and return the full response.

    Uses non-streaming mode. 

    Parameters
    ----------
    model:
        Name of the OpenRouter model (e.g., 'google/gemini-2.5-pro').
    user_prompt:
        The main question or task text.
    system_prompt:
        Optional system instruction.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    tuple[str, int, float]
        ``(response_text, token_count, elapsed_seconds)``
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",  # Required by OpenRouter for ranking
        "X-Title": "LLM Benchmark",
    }

    start = time.monotonic()
    
    with httpx.Client(timeout=timeout) as client:
        resp = client.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()

    elapsed = time.monotonic() - start
    data = resp.json()

    # Safely extract response and token usage
    choices = data.get("choices", [])
    text = ""
    if choices:
        text = choices[0].get("message", {}).get("content", "").strip()
        
    usage = data.get("usage", {})
    tokens: int = usage.get("completion_tokens", 0)

    return text, tokens, round(elapsed, 2)
