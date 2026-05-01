"""LLM client utilities for the project."""

from __future__ import annotations

import os

import requests

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 180


def get_ollama_base_url() -> str:
    """Return the configured Ollama base URL."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()


def get_ollama_model() -> str:
    """Return the configured Ollama model name."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def get_ollama_timeout_seconds() -> int:
    """Return the configured Ollama request timeout."""
    value = os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
    try:
        return max(1, int(value))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS


def build_payload(prompt: str) -> dict[str, object]:
    """Build the Ollama request payload for a prompt."""
    return {
        "model": get_ollama_model(),
        "prompt": prompt,
        "stream": False,
    }


def generate_response(prompt: str) -> str:
    """Send a prompt to Ollama and return the generated response."""
    endpoint = f"{get_ollama_base_url().rstrip('/')}/api/generate"

    try:
        response = requests.post(
            endpoint,
            json=build_payload(prompt),
            timeout=get_ollama_timeout_seconds(),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Ollama API error: {exc}"

    try:
        data = response.json()
    except ValueError:
        return "Ollama API error: invalid JSON response received from the server."

    generated_text = str(data.get("response", "")).strip()
    if generated_text:
        return generated_text

    return "Ollama API error: empty response received from the model."


def is_error_response(response: str) -> bool:
    """Return True when the model call returned a readable error string."""
    return response.startswith("Ollama API error:")
