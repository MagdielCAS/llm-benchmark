"""
ollama_client.py — Thin wrapper around the Ollama local REST API.

Exposes three functions used by the rest of the benchmark suite:

* :func:`check_running` – quick health-check before the run starts.
* :func:`list_models`   – fetch all locally available model names.
* :func:`generate`      – send a prompt and return the full response text.

All HTTP calls use ``httpx`` with conservative timeouts appropriate for
local inference on an M1 MacBook Pro (16 GB RAM).
"""

import time
import httpx
from typing import Optional
from config import OLLAMA_BASE_URL


def list_models() -> list[str]:
    """Return a sorted list of locally available Ollama model names.

    Calls ``GET /api/tags`` and extracts the ``name`` field from each
    entry in the ``models`` array.

    Returns
    -------
    list[str]
        Model names such as ``['llama3.2', 'mistral:7b']``.

    Raises
    ------
    httpx.HTTPError
        If Ollama is not reachable or returns a non-2xx status.
    """
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return sorted(m["name"] for m in data.get("models", []))


def generate(
    model: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    timeout: float = 300.0,
) -> tuple[str, int, float]:
    """Call ``POST /api/generate`` and return the complete model response.

    Uses non-streaming mode so the full response is received atomically.
    Wall-clock time is measured with ``time.monotonic`` for accuracy
    independent of system clock adjustments.

    Parameters
    ----------
    model:
        Name of the Ollama model to query (e.g. ``'llama3.2'``).
    user_prompt:
        The main question or task text shown to the model.
    system_prompt:
        Optional system instruction prepended to the conversation.
    timeout:
        HTTP request timeout in seconds (default 300 s — generous for
        large models on CPU/limited VRAM).

    Returns
    -------
    tuple[str, int, float]
        ``(response_text, token_count, elapsed_seconds)``

    Raises
    ------
    httpx.HTTPError
        On non-2xx responses or network failures.
    """
    payload: dict = {
        "model": model,
        "prompt": user_prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    start = time.monotonic()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        resp.raise_for_status()

    elapsed = time.monotonic() - start
    data = resp.json()
    text: str = data.get("response", "").strip()
    tokens: int = data.get("eval_count", 0)
    return text, tokens, round(elapsed, 2)


def check_running() -> bool:
    """Return ``True`` if the Ollama server is reachable.

    Performs a lightweight ``GET /api/tags`` with a short timeout.  Any
    exception (connection refused, timeout, etc.) is caught and returns
    ``False`` rather than propagating — callers should surface a friendly
    error message instead.
    """
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
