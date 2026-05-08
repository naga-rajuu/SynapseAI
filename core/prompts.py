"""Centralized LangChain prompt templates for agent nodes."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.agent_factory import AgentSpec


def manager_plan_prompt() -> ChatPromptTemplate:
    """Prompt for the manager to decide which teams are needed."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior engineering manager for an autonomous software organization. "
                "Decide which teams are required for the request and give each needed team a concise task. "
                "Be selective and do not activate unnecessary teams.\n{format_instructions}",
            ),
            (
                "human",
                "Project request:\n{project_request}\n\n"
                "Available teams:\n{team_roster}",
            ),
        ]
    )


def lead_plan_prompt(spec: AgentSpec) -> ChatPromptTemplate:
    """Prompt for a team lead to create independent worker assignments."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are {spec.name}, a {spec.team} team lead. "
                "Create at most four independent worker assignments. "
                "Complex work goes to senior workers, simpler work goes to junior workers. "
                "Do not duplicate files across workers. Mark workers idle if they are not needed.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Project request:\n{project_request}\n\n"
                "Team task:\n{team_task}\n\n"
                "Workers:\n{worker_roster}\n\n"
                "Candidate work items:\n{candidate_work}",
            ),
        ]
    )


def worker_execution_prompt(spec: AgentSpec) -> ChatPromptTemplate:
    """Prompt for a worker's execution summary."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are {spec.name}, a {spec.seniority} {spec.team} engineer. "
                f"Your specialty is {spec.focus}. "
                "Return a concise execution summary for only your assigned task.",
            ),
            (
                "human",
                "Project request:\n{project_request}\n\n"
                "Team task:\n{team_task}\n\n"
                "Assigned subtask:\n{subtask}\n\n"
                "Owned files:\n{planned_files}",
            ),
        ]
    )


def review_prompt(spec: AgentSpec) -> ChatPromptTemplate:
    """Prompt for structured single-worker lead review."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are {spec.name}, a strict {spec.team} lead reviewer. "
                "Review one completed worker change for correctness, duplication, naming, "
                "maintainability, consistency, integration fit, and edge cases.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Project request:\n{project_request}\n\n"
                "Team task:\n{team_task}\n\n"
                "Worker:\n{worker_name}\n\n"
                "Subtask:\n{subtask}\n\n"
                "Planned files:\n{planned_files}\n\n"
                "Worker output:\n{worker_output}",
            ),
        ]
    )


def merge_prompt(spec: AgentSpec) -> ChatPromptTemplate:
    """Prompt for team-level output merge."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are {spec.name}. Merge approved worker outputs into a concise team delivery summary. "
                "Return 3 to 5 practical lines with no intro.",
            ),
            (
                "human",
                "Team task:\n{team_task}\n\n"
                "Approved worker outputs:\n{approved_outputs}",
            ),
        ]
    )


def integration_prompt() -> ChatPromptTemplate:
    """Prompt for final organization-wide integration."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an integration agent. Create a concise final engineering summary from merged team outputs. "
                "Keep it practical and under 8 lines.",
            ),
            (
                "human",
                "Project request:\n{project_request}\n\n"
                "Merged team outputs:\n{merged_output}",
            ),
        ]
    )        


def render_prompt_messages(
    prompt: ChatPromptTemplate,
    variables: dict[str, object],
) -> list[BaseMessage]:
    """Render a prompt template into a concrete message list."""
    return list(prompt.format_messages(**variables))


def build_message_history(
    prompt: ChatPromptTemplate,
    variables: dict[str, object],
    prior_ai_context: str | None = None,
    prior_messages: Sequence[BaseMessage] | None = None,
    follow_up_human_input: str | None = None,
) -> list[BaseMessage]:
    """Build a compact message history with optional prior context."""
    rendered = render_prompt_messages(prompt, variables)
    messages: list[BaseMessage] = []
    for message in rendered:
        if isinstance(message, SystemMessage):
            messages.append(SystemMessage(content=message.content))
        elif isinstance(message, HumanMessage):
            messages.append(HumanMessage(content=message.content))
        elif isinstance(message, AIMessage):
            messages.append(AIMessage(content=message.content))
        else:
            messages.append(message)
    if prior_messages:
        messages.extend(prior_messages)
    if prior_ai_context:
        messages.append(AIMessage(content=prior_ai_context))
    if follow_up_human_input:
        messages.append(HumanMessage(content=follow_up_human_input))
    return messages
