"""Centralized LangChain prompt templates for agent nodes."""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.agent_factory import AgentSpec


def intent_router_prompt() -> ChatPromptTemplate:
    """Prompt for generic-vs-project intent classification."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You route user requests for a software engineering platform. "
                "Classify each request into exactly one type: GENERIC_CHAT or PROJECT_RELATED. "
                "Choose PROJECT_RELATED for build requests, modification requests, repository questions, architecture questions, or any software project workflow.\n"
                "{format_instructions}",
            ),
            ("human", "User request:\n{project_request}"),
        ]
    )


def generic_response_prompt() -> ChatPromptTemplate:
    """Prompt for normal generic chat responses."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise, helpful assistant."),
            ("human", "{project_request}"),
        ]
    )


def onboarding_manager_prompt() -> ChatPromptTemplate:
    """Prompt for collecting only missing onboarding details."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a project onboarding manager. Collect only missing GitHub and project details. "
                "Be concise and professional. Ask only for fields that are missing or invalid.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Original request:\n{project_request}\n\n"
                "Known onboarding details:\n{known_details}\n\n"
                "Missing fields:\n{missing_fields}\n\n"
                "Validation errors:\n{validation_errors}",
            ),
        ]
    )


def manager_plan_prompt() -> ChatPromptTemplate:
    """Prompt for the manager to decide execution strategy and teams after onboarding."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior engineering manager. The GitHub onboarding phase is complete and the repository is ready. "
                "Decide whether the request is BUILD_PROJECT, MODIFY_PROJECT, or PROJECT_QUERY. "
                "If it is PROJECT_QUERY, avoid unnecessary engineering team activation. "
                "For build or modify work, choose only the required teams and define concise team tasks.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Original request:\n{project_request}\n\n"
                "Repository context:\n{repo_context}\n\n"
                "Gathered requirements:\n{requirements}\n\n"
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
                f"You are {spec.name}, a {spec.team} technical lead. "
                "Create at most four independent subtasks. "
                "Complex work goes to senior developers. Simpler scoped work goes to junior developers. "
                "Do not hardcode filenames. Describe outcomes and responsibilities only. Mark unused workers idle.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Execution mode: {execution_mode}\n\n"
                "Project: {active_project}\n\n"
                "Original request:\n{project_request}\n\n"
                "Team task:\n{team_task}\n\n"
                "Workers:\n{worker_roster}\n\n"
                "Candidate work areas:\n{candidate_work}",
            ),
        ]
    )


def worker_execution_prompt(spec: AgentSpec) -> ChatPromptTemplate:
    """Prompt for a worker execution summary."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are {spec.name}, a {spec.seniority} {spec.team} engineer. "
                f"Your specialty is {spec.focus}. "
                "Return a concise execution summary for only your assigned subtask.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Execution mode: {execution_mode}\n\n"
                "Project: {active_project}\n\n"
                "Original request:\n{project_request}\n\n"
                "Team task:\n{team_task}\n\n"
                "Assigned subtask:\n{subtask}\n\n"
                "Expected outcome:\n{expected_outcome}",
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
                "Review one completed worker output for correctness, scope fit, maintainability, consistency, and edge cases.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Execution mode: {execution_mode}\n\n"
                "Project: {active_project}\n\n"
                "Original request:\n{project_request}\n\n"
                "Team task:\n{team_task}\n\n"
                "Worker:\n{worker_name}\n\n"
                "Subtask:\n{subtask}\n\n"
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
                "Return 3 to 5 practical lines.",
            ),
            (
                "human",
                "Team task:\n{team_task}\n\n"
                "Approved worker outputs:\n{approved_outputs}",
            ),
        ]
    )


def project_analyst_prompt() -> ChatPromptTemplate:
    """Prompt for repository explanation mode."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior project analyst for a software engineering repository. "
                "Answer repository-specific questions clearly and accurately. Cite relevant files and components.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "User question:\n{project_request}\n\n"
                "Repository snapshot:\n{repo_snapshot}",
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
                "Execution mode: {execution_mode}\n\n"
                "Project: {active_project}\n\n"
                "Original request:\n{project_request}\n\n"
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
