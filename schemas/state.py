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
    worker_model_map: dict[str, str]
    lead_statuses: Annotated[dict[str, dict[str, str]], operator.or_]
    worker_statuses: Annotated[dict[str, dict[str, str]], operator.or_]
    worker_assignments: Annotated[dict[str, dict[str, object]], operator.or_]
    review_comments: Annotated[dict[str, dict[str, object]], operator.or_]
    approved_outputs: Annotated[dict[str, dict[str, object]], operator.or_]
    review_queues: Annotated[dict[str, list[str]], operator.or_]
    reviewed_paths: Annotated[dict[str, list[str]], operator.or_]
    idle_workers: Annotated[dict[str, list[str]], operator.or_]
    assigned_files: Annotated[dict[str, str], operator.or_]
    created_files: Annotated[dict[str, str], operator.or_]
    file_owner: Annotated[dict[str, str], operator.or_]
    final_team_outputs: Annotated[dict[str, dict[str, str]], operator.or_]
    team_review_rounds: Annotated[dict[str, int], operator.or_]
    worker_attempts: Annotated[dict[str, int], operator.or_]
    tool_call_records: Annotated[list[dict[str, object]], operator.add]
    worker_outputs: Annotated[dict[str, dict[str, object]], operator.or_]
    lead_outputs: Annotated[list[dict[str, str]], operator.add]
    audit_log_path: str
    merged_output: str
    final_output: str
    active_team: str
    active_worker: str
