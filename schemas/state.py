"""Shared graph state definitions."""

from __future__ import annotations

from typing import TypedDict


class WorkflowState(TypedDict):
    """State shared across the LangGraph workflow."""

    user_input: str
    task_breakdown: str
    dev_output: str
    final_output: str
