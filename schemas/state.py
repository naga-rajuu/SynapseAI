"""Shared graph state definitions."""

from __future__ import annotations

import operator
from typing import Annotated
from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    """State shared across the LangGraph workflow."""

    project_request: str
    team_tasks: dict[str, dict[str, object]]
    project_status: str
    lead_statuses: Annotated[dict[str, dict[str, str]], operator.or_]
    worker_statuses: Annotated[dict[str, dict[str, dict[str, str]]], operator.or_]
    worker_outputs: Annotated[list[dict[str, str]], operator.add]
    lead_outputs: Annotated[list[dict[str, str]], operator.add]
    merged_output: str
    final_output: str
