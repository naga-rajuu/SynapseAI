"""Structured LLM output models used by agent nodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class TeamDecision(BaseModel):
    """One team decision from the manager."""

    needed: bool
    task: str = ""


class ManagerPlan(BaseModel):
    """Structured manager output across all engineering teams."""

    backend: TeamDecision
    frontend: TeamDecision
    qa: TeamDecision
    devops: TeamDecision


class LeadAssignment(BaseModel):
    """One worker assignment emitted by a lead."""

    worker_name: str
    subtask: str
    planned_files: list[str] = Field(default_factory=list)
    complexity: Literal["complex", "simple"] = "simple"
    idle: bool = False


class LeadPlan(BaseModel):
    """Structured lead planning output."""

    assignments: list[LeadAssignment] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    """One lead review decision for a worker output."""

    worker_name: str
    status: Literal["approved", "needs_fix"]
    note: str
    comments: list[str] = Field(default_factory=list)


class ReviewBatch(BaseModel):
    """Structured set of review decisions for one queue drain."""

    decisions: list[ReviewDecision] = Field(default_factory=list)
