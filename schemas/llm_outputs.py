"""Structured LLM output models used by graph nodes and agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field

RequestType = Literal["GENERIC_CHAT", "PROJECT_RELATED"]
ExecutionMode = Literal["BUILD_PROJECT", "MODIFY_PROJECT", "PROJECT_QUERY"]
TaskComplexity = Literal["complex", "simple"]
TaskPriority = Literal["high", "medium", "low"]
ReviewStatus = Literal["approved", "needs_fix"]


class IntentClassification(BaseModel):
    """Route classification for an incoming user request."""

    request_type: RequestType
    rationale: str


class OnboardingQuestionSet(BaseModel):
    """Concise follow-up questions for missing onboarding details."""

    message: str


class ValidationResult(BaseModel):
    """Repository onboarding validation result."""

    status: Literal["passed", "failed"]
    missing: list[str] = Field(default_factory=list)
    invalid: list[str] = Field(default_factory=list)
    message: str = ""
    repo_ready: bool = False


class RepoPlan(BaseModel):
    """Repository workflow intent produced by the manager."""

    use_current_repo: bool = True
    ensure_main_branch: bool = True
    sync_with_remote: bool = True
    project_branch: str = ""
    notes: str = ""


class TeamExecutionDecision(BaseModel):
    """One team decision from the manager."""

    needed: bool
    priority: TaskPriority = "medium"
    task: str = ""


class ManagerPlan(BaseModel):
    """Structured manager output across all engineering teams."""

    execution_mode: ExecutionMode
    active_project: str
    execution_summary: str
    backend: TeamExecutionDecision
    frontend: TeamExecutionDecision
    qa: TeamExecutionDecision
    devops: TeamExecutionDecision
    repo_plan: RepoPlan = Field(default_factory=RepoPlan)


class LeadAssignment(BaseModel):
    """One worker assignment emitted by a lead."""

    worker_name: str
    subtask: str
    expected_outcome: str = ""
    complexity: TaskComplexity = "simple"
    idle: bool = False


class LeadPlan(BaseModel):
    """Structured lead planning output."""

    assignments: list[LeadAssignment] = Field(default_factory=list)


class WorkerDelivery(BaseModel):
    """Structured worker delivery summary."""

    summary: str
    commit_message: str
    implementation_notes: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    """One lead review decision for a worker output."""

    worker_name: str
    status: ReviewStatus
    note: str
    comments: list[str] = Field(default_factory=list)


class ReviewBatch(BaseModel):
    """Structured set of review decisions for one queue drain."""

    decisions: list[ReviewDecision] = Field(default_factory=list)


class AnalystAnswer(BaseModel):
    """Structured answer for project-query mode."""

    answer: str
    relevant_files: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
