"""Engineering lead, worker, and integration agents."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
from typing import Any
from typing import Iterable

from agents.agent_factory import AgentSpec
from core.llm import generate_response
from core.llm import is_error_response
from tools.approvals import ApprovalManager
from tools.context import ToolExecutionContext
from tools.permissions import get_allowed_tools
from tools.tool_intelligence import infer_app_name
from tools.tool_intelligence import infer_tool_requests
from tools.tool_router import execute_tool

MAX_WORKERS_PER_TEAM = 4
MAX_REVIEW_ROUNDS = 1


class TeamLeadAgent:
    """Lead agent responsible for splitting, reviewing, and merging team work."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def plan_worker_tasks(
        self,
        project_request: str,
        team_task: str,
        workers: list[AgentSpec],
    ) -> dict[str, Any]:
        """Split work into independent subtasks with explicit file ownership."""
        worker_pool = workers[:MAX_WORKERS_PER_TEAM]
        subtasks = self.build_default_subtasks(project_request, team_task)[: len(worker_pool)]

        senior_workers = [worker for worker in worker_pool if worker.seniority == "senior"]
        junior_workers = [worker for worker in worker_pool if worker.seniority != "senior"]
        assignments: dict[str, dict[str, object]] = {}
        idle_workers = [worker.name for worker in worker_pool]
        assigned_files: dict[str, str] = {}
        team_files = [path for item in subtasks for path in item["planned_files"]]

        for index, task in enumerate(subtasks):
            worker = self.select_worker_for_task(task["complexity"], senior_workers, junior_workers)
            if worker is None:
                continue
            if worker.name in idle_workers:
                idle_workers.remove(worker.name)
            assignments[worker.key] = {
                "worker_name": worker.name,
                "team": worker.team,
                "subtask": task["subtask"],
                "planned_files": task["planned_files"],
                "team_files": team_files,
                "complexity": task["complexity"],
                "setup_folders": sorted(
                    {
                        str(PurePosixPath(path).parent)
                        for path in task["planned_files"]
                        if str(PurePosixPath(path).parent) not in {"", "."}
                    }
                )
                if index == 0
                else [],
            }
            for path in task["planned_files"]:
                assigned_files[path] = worker.name

        return {
            "assignments": assignments,
            "idle_workers": idle_workers,
            "assigned_files": assigned_files,
        }

    def review_worker_outputs(
        self,
        project_request: str,
        team_task: str,
        assignments: dict[str, dict[str, object]],
        worker_outputs: dict[str, dict[str, object]],
        review_round: int,
        reviewed_paths: list[str] | None = None,
    ) -> dict[str, dict[str, object]]:
        """Review worker outputs and route only affected workers back for fixes."""
        review_queue = self.build_review_queue(assignments, worker_outputs)
        reviewed_set = {normalize_path(path) for path in (reviewed_paths or [])}
        decisions: dict[str, dict[str, object]] = {}

        for worker_key in review_queue:
            assignment = {worker_key: assignments[worker_key]}
            output = {worker_key: worker_outputs[worker_key]}
            duplicate_comment = detect_duplicate_review_work(
                worker_outputs[worker_key],
                reviewed_set,
            )
            if duplicate_comment:
                if review_round < MAX_REVIEW_ROUNDS:
                    decisions[worker_key] = {
                        "status": "needs_fix",
                        "comments": [duplicate_comment],
                        "note": "Avoid repeating already reviewed work.",
                    }
                else:
                    decisions[worker_key] = {
                        "status": "approved",
                        "comments": [duplicate_comment],
                        "note": "Approved with duplication warning recorded.",
                    }
                continue

            parsed = self.parse_review_response(
                generate_response(
                    self.build_single_review_prompt(
                        project_request=project_request,
                        team_task=team_task,
                        assignment=assignments[worker_key],
                        worker_output=worker_outputs[worker_key],
                    ),
                    model=self.spec.model,
                ),
                assignment,
            )
            if parsed:
                decisions[worker_key] = parsed[worker_key]
            else:
                fallback = self.build_review_fallback(assignment, output, review_round)
                decisions[worker_key] = fallback[worker_key]

            if str(decisions[worker_key]["status"]) == "approved":
                reviewed_set.update(
                    normalize_path(path)
                    for path in worker_outputs[worker_key].get("created_paths", [])
                )

        return normalize_review_decisions(decisions, review_round)

    def build_review_queue(
        self,
        assignments: dict[str, dict[str, object]],
        worker_outputs: dict[str, dict[str, object]],
    ) -> list[str]:
        """Build the queue of finished worker outputs waiting for lead review."""
        queued = []
        for worker_key, assignment in assignments.items():
            if worker_key not in worker_outputs:
                continue
            complexity = str(assignment.get("complexity", "simple"))
            priority = 0 if complexity == "complex" else 1
            queued.append((priority, str(assignment["worker_name"]), worker_key))
        queued.sort()
        return [worker_key for _, _, worker_key in queued]

    def build_single_review_prompt(
        self,
        project_request: str,
        team_task: str,
        assignment: dict[str, object],
        worker_output: dict[str, object],
    ) -> str:
        """Build a focused review prompt for a single completed worker change."""
        planned_files = [str(path) for path in assignment.get("planned_files", [])]
        return (
            f"You are {self.spec.name}.\n"
            "Review this single worker result for correctness, duplication, naming, "
            "maintainability, consistency, integration fit, and edge cases.\n"
            "Return exactly one line in this format:\n"
            "Worker Name|APPROVED|short note\n"
            "or\n"
            "Worker Name|REVISE|specific fix comment\n"
            "No extra commentary.\n\n"
            f"Project request: {project_request}\n"
            f"Team task: {team_task}\n"
            f"Worker: {assignment['worker_name']}\n"
            f"Subtask: {assignment['subtask']}\n"
            f"Planned files: {', '.join(planned_files)}\n"
            f"Worker output: {worker_output.get('output', '')}"
        )

    def build_review_prompt(
        self,
        project_request: str,
        team_task: str,
        assignments: dict[str, dict[str, object]],
        worker_outputs: dict[str, dict[str, object]],
    ) -> str:
        """Build a concise review prompt for the lead."""
        worker_lines = []
        for worker_key, assignment in assignments.items():
            result = worker_outputs.get(worker_key, {})
            worker_lines.append(
                f"{assignment['worker_name']} | task={assignment['subtask']} | "
                f"files={', '.join(assignment['planned_files'])} | output={result.get('output', '')}"
            )
        worker_block = "\n".join(worker_lines)
        return (
            f"You are {self.spec.name}.\n"
            "Review each worker result for correctness, duplication, naming, maintainability, "
            "integration fit, and missing edge cases.\n"
            "Return exactly one line per assigned worker using this format:\n"
            "Worker Name|APPROVED|short note\n"
            "or\n"
            "Worker Name|REVISE|specific fix comment\n"
            "No extra commentary.\n\n"
            f"Project request: {project_request}\n"
            f"Team task: {team_task}\n"
            f"Worker outputs:\n{worker_block}"
        )

    def parse_review_response(
        self,
        raw_response: str,
        assignments: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        """Parse targeted review decisions from the lead LLM output."""
        if is_error_response(raw_response):
            return {}
        name_to_key = {
            str(assignment["worker_name"]): worker_key
            for worker_key, assignment in assignments.items()
        }
        decisions: dict[str, dict[str, object]] = {}
        for line in raw_response.splitlines():
            parts = [part.strip() for part in line.split("|", 2)]
            if len(parts) != 3:
                continue
            worker_name, status, note = parts
            worker_key = name_to_key.get(worker_name)
            if not worker_key:
                continue
            decisions[worker_key] = {
                "status": "approved" if status.upper() == "APPROVED" else "needs_fix",
                "comments": [] if status.upper() == "APPROVED" else [note],
                "note": note,
            }
        return decisions if len(decisions) == len(assignments) else {}

    def build_review_fallback(
        self,
        assignments: dict[str, dict[str, object]],
        worker_outputs: dict[str, dict[str, object]],
        review_round: int,
    ) -> dict[str, dict[str, object]]:
        """Review worker outputs without relying on model formatting."""
        decisions: dict[str, dict[str, object]] = {}
        for worker_key, assignment in assignments.items():
            output = worker_outputs.get(worker_key, {})
            tool_calls = output.get("tool_calls", [])
            planned_files = [str(path) for path in assignment.get("planned_files", [])]
            failure_reasons = detect_worker_issues(output, planned_files)
            if failure_reasons and review_round < MAX_REVIEW_ROUNDS:
                decisions[worker_key] = {
                    "status": "needs_fix",
                    "comments": failure_reasons,
                    "note": "Needs targeted fixes before approval.",
                }
            else:
                decisions[worker_key] = {
                    "status": "approved",
                    "comments": [],
                    "note": summarize_tool_outcome(tool_calls) or "Approved.",
                }
        return decisions

    def merge_worker_outputs(
        self,
        team_task: str,
        approved_outputs: dict[str, dict[str, object]],
    ) -> str:
        """Merge approved worker outputs into one team summary."""
        output_lines = "\n".join(
            f"{item['worker_name']}: {item['output']}"
            for item in approved_outputs.values()
        )
        merged = generate_response(
            (
                f"You are {self.spec.name}.\n"
                "Merge approved worker outputs into a concise team delivery summary.\n"
                "Return 3 to 5 short lines. No intro.\n\n"
                f"Team task: {team_task}\n"
                f"Approved worker outputs:\n{output_lines}"
            ),
            model=self.spec.model,
        )
        if is_error_response(merged):
            return "\n".join(
                f"- {item['worker_name']}: {item['output']}"
                for item in approved_outputs.values()
            )
        return merged

    def select_worker_for_task(
        self,
        complexity: str,
        senior_workers: list[AgentSpec],
        junior_workers: list[AgentSpec],
    ) -> AgentSpec | None:
        """Assign complex work to seniors first and simpler work to juniors."""
        if complexity == "complex" and senior_workers:
            return senior_workers.pop(0)
        if complexity == "simple" and junior_workers:
            return junior_workers.pop(0)
        if senior_workers:
            return senior_workers.pop(0)
        if junior_workers:
            return junior_workers.pop(0)
        return None

    def build_default_subtasks(self, project_request: str, team_task: str) -> list[dict[str, object]]:
        """Build independent subtasks with unique file ownership."""
        app_name = infer_app_name(project_request)
        if self.spec.team == "frontend":
            base = f"generated_apps/{app_name}"
            subtasks = [
                {
                    "subtask": "Build client-side state and interaction logic.",
                    "complexity": "complex",
                    "planned_files": [f"{base}/app.js"],
                },
                {
                    "subtask": "Create accessible markup and reusable button structure.",
                    "complexity": "complex",
                    "planned_files": [f"{base}/index.html"],
                },
                {
                    "subtask": "Create responsive styling and visual polish.",
                    "complexity": "simple",
                    "planned_files": [f"{base}/styles.css"],
                },
            ]
            if "dashboard" in team_task.lower() or "app" in project_request.lower():
                subtasks.append(
                    {
                        "subtask": "Document UI structure and usage notes.",
                        "complexity": "simple",
                        "planned_files": [f"{base}/README.md"],
                    }
                )
            return subtasks

        if self.spec.team == "backend":
            base = f"generated_apps/{app_name}_backend"
            return [
                {
                    "subtask": "Design service boundaries and shared business logic.",
                    "complexity": "complex",
                    "planned_files": [f"{base}/services.py"],
                },
                {
                    "subtask": "Define API schemas and integration contracts.",
                    "complexity": "complex",
                    "planned_files": [f"{base}/schemas.py"],
                },
                {
                    "subtask": "Implement the primary API entrypoint.",
                    "complexity": "simple",
                    "planned_files": [f"{base}/app.py"],
                },
                {
                    "subtask": "Document endpoint behavior and validation notes.",
                    "complexity": "simple",
                    "planned_files": [f"{base}/README.md"],
                },
            ]

        if self.spec.team == "qa":
            base = f"generated_apps/{app_name}_qa"
            return [
                {
                    "subtask": "Define risk-focused quality strategy and review coverage.",
                    "complexity": "complex",
                    "planned_files": [f"{base}/qa_strategy.md"],
                },
                {
                    "subtask": "Create smoke-test and regression checklist.",
                    "complexity": "simple",
                    "planned_files": [f"{base}/smoke_checklist.md"],
                },
            ]

        base = f"generated_apps/{app_name}_ops"
        return [
            {
                "subtask": "Prepare container and runtime packaging strategy.",
                "complexity": "complex",
                "planned_files": [f"{base}/Dockerfile"],
            },
            {
                "subtask": "Set up pipeline and release automation notes.",
                "complexity": "simple",
                "planned_files": [f"{base}/ci.yml"],
            },
        ]


class WorkerAgent:
    """Worker agent responsible for a local execution summary."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def build_prompt(
        self,
        project_request: str,
        team_task: str,
        subtask: str,
        planned_files: list[str],
        review_comments: list[str] | None = None,
    ) -> str:
        role_focus = (
            "architecture, optimization, and review quality"
            if self.spec.seniority == "senior"
            else "implementation, boilerplate, and practical execution"
        )
        comment_block = ""
        if review_comments:
            comment_block = "Lead review comments:\n" + "\n".join(f"- {item}" for item in review_comments)
        files_block = ", ".join(planned_files) if planned_files else "None"
        return (
            f"You are {self.spec.name}.\n"
            f"Team: {self.spec.team}.\n"
            f"Focus on {role_focus}.\n"
            "Respond with a concise execution summary for only your subtask.\n"
            "Keep it under 5 lines and avoid repeating the full project context.\n\n"
            f"Project request: {project_request}\n"
            f"Team task: {team_task}\n"
            f"Your subtask: {subtask}\n"
            f"Owned files: {files_block}\n"
            f"{comment_block}"
        )

    def run(
        self,
        project_request: str,
        team_task: str,
        subtask: str,
        planned_files: list[str],
        review_comments: list[str] | None = None,
        model: str | None = None,
    ) -> str:
        """Return the worker output for a local subtask."""
        output = generate_response(
            self.build_prompt(project_request, team_task, subtask, planned_files, review_comments),
            model=model or self.spec.model,
        )
        if is_error_response(output):
            return f"Execution fallback: {subtask}"
        return output

    def execute(
        self,
        project_request: str,
        team_task: str,
        assignment: dict[str, object],
        file_owner: dict[str, str],
        created_files: dict[str, str],
        approval_manager: ApprovalManager | None = None,
        model: str | None = None,
        review_comments: list[str] | None = None,
    ) -> dict[str, object]:
        """Run the local worker task and execute any inferred tool requests."""
        planned_files = [str(item) for item in assignment.get("planned_files", [])]
        team_files = [str(item) for item in assignment.get("team_files", planned_files)]
        setup_folders = [str(item) for item in assignment.get("setup_folders", [])]
        subtask = str(assignment["subtask"])
        context = ToolExecutionContext(
            agent_name=self.spec.name,
            role=self.spec.role,
            team=self.spec.team,
            seniority=self.spec.seniority,
            model=(model or self.spec.model or ""),
        )
        allowed_tools = sorted(get_allowed_tools(context))
        tool_requests = infer_tool_requests(
            project_request=project_request,
            team_task=team_task,
            subtask=subtask,
            allowed_tools=allowed_tools,
            assigned_files=planned_files,
            all_team_files=team_files,
            setup_folders=setup_folders,
        )
        tool_results = [
            execute_tool(
                tool_name=item["tool"],
                params=item["params"],
                context=context,
                approval_manager=approval_manager,
                assigned_files=tuple(planned_files),
                file_owner=file_owner,
                created_files=created_files,
            )
            for item in tool_requests
        ]
        tool_summary = build_tool_summary(tool_results)
        output = self.run(
            project_request=project_request,
            team_task=team_task,
            subtask=subtask,
            planned_files=planned_files,
            review_comments=review_comments,
            model=model,
        )
        if tool_summary:
            output = f"{output}\nTool usage: {tool_summary}"
        return {
            "team": self.spec.team,
            "worker_key": self.spec.key,
            "worker_name": self.spec.name,
            "seniority": self.spec.seniority,
            "subtask": subtask,
            "planned_files": planned_files,
            "output": output,
            "tool_calls": tool_results,
            "created_files": collect_created_files(tool_results, self.spec.name),
            "created_paths": collect_created_paths(tool_results),
        }


class IntegrationAgent:
    """Final integration agent for cross-team synthesis."""

    def build_prompt(self, project_request: str, merged_output: str) -> str:
        return (
            "You are an integration agent.\n"
            "Create a concise final engineering summary from the merged team outputs.\n"
            "Keep it practical and under 8 lines.\n\n"
            f"Project request: {project_request}\n"
            f"Merged team outputs:\n{merged_output}"
        )

    def run(self, project_request: str, merged_output: str) -> str:
        """Generate the final integration summary."""
        output = generate_response(self.build_prompt(project_request, merged_output))
        if is_error_response(output):
            return merged_output
        return output


def build_tool_summary(tool_results: list[dict[str, object]]) -> str:
    """Build a short summary of tool usage for worker output."""
    if not tool_results:
        return ""
    summaries = []
    for item in tool_results:
        if bool(item["success"]):
            summaries.append(f"{item['tool']} succeeded")
        elif bool(item.get("approval_required")):
            summaries.append(f"{item['tool']} needs approval")
        else:
            summaries.append(f"{item['tool']} failed")
    return ", ".join(summaries)


def collect_created_files(
    tool_results: list[dict[str, object]],
    owner_name: str,
) -> dict[str, str]:
    """Collect file paths touched successfully by file-writing tools."""
    created: dict[str, str] = {}
    for item in tool_results:
        if not bool(item.get("success")):
            continue
        if item["tool"] not in {"write_file", "append_file", "edit_file"}:
            continue
        for path in item.get("touched_paths", []):
            created[str(path)] = owner_name
    return created


def collect_created_paths(tool_results: list[dict[str, object]]) -> list[str]:
    """Collect all successful touched paths for sequential lead review."""
    created: list[str] = []
    for item in tool_results:
        if not bool(item.get("success")):
            continue
        for path in item.get("touched_paths", []):
            normalized = normalize_path(str(path))
            if normalized not in created:
                created.append(normalized)
    return created


def detect_worker_issues(output: dict[str, object], planned_files: list[str]) -> list[str]:
    """Detect issues that should trigger a fix loop."""
    issues: list[str] = []
    text = str(output.get("output", ""))
    lowered = text.lower()
    if not text or "execution fallback" in lowered or "ollama api error" in lowered:
        issues.append("Provide a stronger implementation summary for the assigned subtask.")

    tool_calls = output.get("tool_calls", [])
    if planned_files and not has_successful_write(tool_calls, planned_files):
        issues.append("Create or update the assigned files before requesting approval.")

    failed_tools = [item for item in tool_calls if not bool(item.get("success"))]
    if failed_tools:
        issues.append("Resolve the failed tool request and keep work within the assigned files.")

    return issues


def has_successful_write(tool_calls: object, planned_files: list[str]) -> bool:
    """Return True when the worker successfully touched one of its assigned files."""
    if not isinstance(tool_calls, list):
        return False
    normalized_files = {normalize_path(path) for path in planned_files}
    for item in tool_calls:
        if not isinstance(item, dict) or not bool(item.get("success")):
            continue
        if item.get("tool") not in {"write_file", "append_file", "edit_file"}:
            continue
        touched = {normalize_path(path) for path in item.get("touched_paths", [])}
        if touched & normalized_files:
            return True
    return False


def summarize_tool_outcome(tool_calls: object) -> str:
    """Summarize a list of tool calls for a review note."""
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    successful = [str(item["tool"]) for item in tool_calls if isinstance(item, dict) and bool(item.get("success"))]
    if not successful:
        return ""
    return f"Approved after {', '.join(successful)}."


def normalize_path(path: str) -> str:
    """Normalize a relative path for consistent comparisons."""
    return str(PurePosixPath(path.replace("\\", "/"))).lstrip("./")


def detect_duplicate_review_work(
    output: dict[str, object],
    reviewed_paths: set[str],
) -> str | None:
    """Detect repeated work against already reviewed changes."""
    for path in output.get("created_paths", []):
        normalized = normalize_path(str(path))
        if normalized in reviewed_paths:
            return f"Do not repeat already reviewed work for {normalized}; reuse the existing artifact."
    return None


def normalize_review_decisions(
    decisions: dict[str, dict[str, object]],
    review_round: int,
) -> dict[str, dict[str, object]]:
    """Cap the review loop so workers do not remain unresolved indefinitely."""
    if review_round < MAX_REVIEW_ROUNDS:
        return decisions

    normalized: dict[str, dict[str, object]] = {}
    for worker_key, decision in decisions.items():
        if str(decision.get("status")) == "needs_fix":
            comments = list(decision.get("comments", []))
            normalized[worker_key] = {
                "status": "approved",
                "comments": comments,
                "note": "Approved after final review round with follow-up notes recorded.",
            }
        else:
            normalized[worker_key] = decision
    return normalized
