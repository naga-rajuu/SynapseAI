"""LangChain-style LLM client utilities for the project."""

from __future__ import annotations

import os
from typing import Any
from typing import Literal
from typing import TypeVar

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
import requests

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 180
LLMRole = Literal["router", "manager", "lead", "worker", "review", "analyst", "integration"]
T = TypeVar("T")


def get_ollama_base_url() -> str:
    """Return the configured Ollama base URL."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()


def get_ollama_model() -> str:
    """Return the configured Ollama model name."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def get_model_for_role(role: LLMRole, override: str | None = None) -> str:
    """Return the configured model for an LLM role with optional override."""
    if override:
        return override.strip()
    env_key = f"{role.upper()}_LLM_MODEL"
    return os.getenv(env_key, get_ollama_model()).strip()


def get_ollama_timeout_seconds() -> int:
    """Return the configured Ollama request timeout."""
    value = os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
    try:
        return max(1, int(value))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SECONDS


class OllamaLangChainLLM:
    """Minimal LangChain-native adapter over the Ollama HTTP generate API."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or get_ollama_model()

    def invoke(
        self,
        input_data: str | ChatPromptValue | list[BaseMessage],
        model: str | None = None,
    ) -> AIMessage:
        """Invoke Ollama with string, prompt value, or message input."""
        messages = to_messages(input_data)
        prompt_text = render_messages(messages)
        response_text = _request_ollama(prompt_text, model=model or self.model)
        return AIMessage(content=response_text)

    def as_runnable(self, model: str | None = None) -> RunnableLambda:
        """Expose the client as a LangChain runnable."""
        return RunnableLambda(lambda value: self.invoke(value, model=model))


def get_llm(model: str | None = None, role: LLMRole | None = None) -> OllamaLangChainLLM:
    """Return the default Ollama-backed LangChain LLM adapter."""
    resolved_model = get_model_for_role(role, override=model) if role else model
    return OllamaLangChainLLM(model=resolved_model)


def invoke_prompt(
    prompt: ChatPromptTemplate,
    variables: dict[str, Any],
    model: str | None = None,
    role: LLMRole | None = None,
) -> str:
    """Invoke a chat prompt template and parse plain text output."""
    llm = get_llm(model=model, role=role)
    chain = prompt | llm.as_runnable(model=llm.model) | StrOutputParser()
    return chain.invoke(variables)


def invoke_structured(
    prompt: ChatPromptTemplate,
    variables: dict[str, Any],
    parser: BaseOutputParser[T],
    model: str | None = None,
    role: LLMRole | None = None,
) -> T:
    """Invoke a prompt and parse structured output."""
    llm = get_llm(model=model, role=role)
    chain = prompt | llm.as_runnable(model=llm.model) | parser
    return chain.invoke(variables)


def invoke_messages(
    messages: list[BaseMessage],
    model: str | None = None,
    role: LLMRole | None = None,
) -> AIMessage:
    """Invoke the model with an explicit message list."""
    llm = get_llm(model=model, role=role)
    return llm.invoke(messages, model=llm.model)


def invoke_messages_text(
    messages: list[BaseMessage],
    model: str | None = None,
    role: LLMRole | None = None,
) -> str:
    """Invoke the model with messages and parse plain text output."""
    response = invoke_messages(messages, model=model, role=role)
    return StrOutputParser().parse(str(response.content))


def invoke_messages_structured(
    messages: list[BaseMessage],
    parser: BaseOutputParser[T],
    model: str | None = None,
    role: LLMRole | None = None,
) -> T:
    """Invoke the model with messages and parse structured output."""
    response = invoke_messages(messages, model=model, role=role)
    return parser.parse(str(response.content))


def generate_response(prompt: str, model: str | None = None) -> str:
    """Compatibility helper for modules that still need plain text generation."""
    message = get_llm(model=model).invoke(prompt, model=model)
    return str(message.content).strip()


def build_payload(prompt: str, model: str | None = None) -> dict[str, object]:
    """Build the Ollama request payload for a prompt."""
    return {
        "model": (model or get_ollama_model()).strip(),
        "prompt": prompt,
        "stream": False,
    }


def to_messages(input_data: str | ChatPromptValue | list[BaseMessage]) -> list[BaseMessage]:
    """Normalize prompt values and strings into LangChain message objects."""
    if isinstance(input_data, list):
        return input_data
    if isinstance(input_data, ChatPromptValue):
        return input_data.to_messages()
    return [HumanMessage(content=str(input_data))]


def render_messages(messages: list[BaseMessage]) -> str:
    """Render messages into a simple chat transcript for Ollama generate."""
    rendered_lines: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "System"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        else:
            role = "Human"
        rendered_lines.append(f"{role}: {message.content}")
    rendered_lines.append("Assistant:")
    return "\n\n".join(rendered_lines)


def _request_ollama(prompt: str, model: str | None = None) -> str:
    """Send a rendered prompt to Ollama and return the response text."""
    endpoint = f"{get_ollama_base_url().rstrip('/')}/api/generate"

    try:
        response = requests.post(
            endpoint,
            json=build_payload(prompt, model=model),
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
