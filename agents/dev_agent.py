"""Engineering lead, worker, and integration agents."""

from __future__ import annotations

from typing import Any

from agents.agent_factory import AgentSpec
from core.llm import invoke_messages_structured
from core.llm import invoke_prompt
from core.llm import invoke_structured
from core.prompts import build_message_history
from core.prompts import integration_prompt
from core.prompts import lead_plan_prompt
from core.prompts import merge_prompt
from core.prompts import review_prompt
from core.prompts import worker_execution_prompt
from langchain_core.output_parsers import PydanticOutputParser
from schemas.llm_outputs import LeadAssignment
from schemas.llm_outputs import LeadPlan
from schemas.llm_outputs import ReviewBatch
from schemas.llm_outputs import WorkerDelivery
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
        execution_mode: str,
        active_project: str,
        team_task: str,
        workers: list[AgentSpec],
    ) -> dict[str, Any]:
        """Split work into independent subtasks without hardcoded filenames."""
        worker_pool = workers[:MAX_WORKERS_PER_TEAM]
        suggested_subtasks = self.build_default_subtasks(project_request, execution_mode, team_task)[: len(worker_pool)]
        assignments = self.generate_structured_assignments(
            project_request=project_request,
            execution_mode=execution_mode,
            active_project=active_project,
            team_task=team_task,
            workers=worker_pool,
            suggested_subtasks=suggested_subtasks,
        )

        result: dict[str, dict[str, object]] = {}
        idle_workers = [worker.name for worker in worker_pool]
        for task in assignments:
            worker_name = str(task["worker_name"])
            worker = next((item for item in worker_pool if item.name == worker_name), None)
            if worker is None or bool(task.get("idle")):
                continue
            if worker.name in idle_workers:
                idle_workers.remove(worker.name)
            result[worker.key] = {
                "worker_name": worker.name,
                "team": worker.team,
                "subtask": task["subtask"],
                "expected_outcome": task["expected_outcome"],
                "complexity": task["complexity"],
            }

        return {"assignments": result, "idle_workers": idle_workers, "assigned_files": {}}

    def generate_structured_assignments(
        self,
        project_request: str,
        execution_mode: str,
        active_project: str,
        team_task: str,
        workers: list[AgentSpec],
        suggested_subtasks: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Use a structured prompt for lead planning."""
        parser = PydanticOutputParser(pydantic_object=LeadPlan)
        try:
            result = invoke_structured(
                prompt=lead_plan_prompt(self.spec),
                variables={
                    "execution_mode": execution_mode,
                    "active_project": active_project,
                    "project_request": project_request,
                    "team_task": team_task,
                    "worker_roster": "\n".join(
                        f"{worker.name} | seniority={worker.seniority} | focus={worker.focus}"
                        for worker in workers
                    ),
                    "candidate_work": "\n".join(
                        f"- subtask={item['subtask']} | complexity={item['complexity']} | outcome={item['expected_outcome']}"
                        for item in suggested_subtasks
                    ),
                    "format_instructions": parser.get_format_instructions(),
                },
                parser=parser,
                role="lead",
                model=self.spec.model,
            )
            normalized = self.normalize_assignments(result.assignments, workers)
            if normalized:
                return normalized
        except Exception:
            pass
        return self.build_fallback_assignments(workers, suggested_subtasks)

    def review_worker_outputs(
        self,
        project_request: str,
        execution_mode: str,
        active_project: str,
        team_task: str,
        assignments: dict[str, dict[str, object]],
        worker_outputs: dict[str, dict[str, object]],
        review_round: int,
    ) -> dict[str, dict[str, object]]:
        """Review worker outputs and route affected workers back for fixes."""
        decisions: dict[str, dict[str, object]] = {}
        for worker_key in self.build_review_queue(assignments, worker_outputs):
            parsed = self.invoke_single_review(
                worker_key=worker_key,
                project_request=project_request,
                execution_mode=execution_mode,
                active_project=active_project,
                team_task=team_task,
                assignment=assignments[worker_key],
                worker_output=worker_outputs[worker_key],
            )
            if parsed:
                decisions[worker_key] = parsed[worker_key]
            else:
                decisions[worker_key] = self.build_review_fallback(assignments[worker_key], worker_outputs[worker_key], review_round)
        return normalize_review_decisions(decisions, review_round)

    def build_review_queue(
        self,
        assignments: dict[str, dict[str, object]],
        worker_outputs: dict[str, dict[str, object]],
    ) -> list[str]:
        queued = []
        for worker_key, assignment in assignments.items():
            if worker_key not in worker_outputs:
                continue
            priority = 0 if str(assignment.get("complexity", "simple")) == "complex" else 1
            queued.append((priority, str(assignment["worker_name"]), worker_key))
        queued.sort()
        return [worker_key for _, _, worker_key in queued]

    def invoke_single_review(
        self,
        worker_key: str,
        project_request: str,
        execution_mode: str,
        active_project: str,
        team_task: str,
        assignment: dict[str, object],
        worker_output: dict[str, object],
    ) -> dict[str, dict[str, object]]:
        parser = PydanticOutputParser(pydantic_object=ReviewBatch)
        messages = build_message_history(
            prompt=review_prompt(self.spec),
            variables={
                "execution_mode": execution_mode,
                "active_project": active_project,
                "project_request": project_request,
                "team_task": team_task,
                "worker_name": str(assignment["worker_name"]),
                "subtask": str(assignment["subtask"]),
                "worker_output": str(worker_output.get("output", "")),
                "format_instructions": parser.get_format_instructions(),
            },
            prior_ai_context=str(worker_output.get("output", "")),
            follow_up_human_input="Review the result above and return the structured decision.",
        )
        try:
            parsed = invoke_messages_structured(messages=messages, parser=parser, role="review", model=self.spec.model)
        except Exception:
            return {}
        return self.parse_review_model(parsed, {worker_key: assignment})

    def parse_review_model(
        self,
        review_batch: ReviewBatch,
        assignments: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        name_to_key = {str(assignment["worker_name"]): worker_key for worker_key, assignment in assignments.items()}
        decisions: dict[str, dict[str, object]] = {}
        for item in review_batch.decisions:
            worker_key = name_to_key.get(item.worker_name)
            if not worker_key:
                continue
            decisions[worker_key] = {"status": item.status, "comments": item.comments, "note": item.note}
        return decisions if len(decisions) == len(assignments) else {}

    def build_fallback_assignments(self, workers: list[AgentSpec], suggested_subtasks: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {
                "worker_name": workers[index].name,
                "subtask": str(item["subtask"]),
                "expected_outcome": str(item["expected_outcome"]),
                "complexity": str(item["complexity"]),
                "idle": False,
            }
            for index, item in enumerate(suggested_subtasks[: len(workers)])
        ]

    def normalize_assignments(self, assignments: list[LeadAssignment], workers: list[AgentSpec]) -> list[dict[str, object]]:
        allowed_workers = {worker.name for worker in workers}
        normalized: list[dict[str, object]] = []
        used_workers: set[str] = set()
        for item in assignments[: len(workers)]:
            if item.worker_name not in allowed_workers or item.worker_name in used_workers:
                continue
            used_workers.add(item.worker_name)
            normalized.append(
                {
                    "worker_name": item.worker_name,
                    "subtask": item.subtask,
                    "expected_outcome": item.expected_outcome,
                    "complexity": item.complexity,
                    "idle": item.idle,
                }
            )
        return normalized

    def build_review_fallback(
        self,
        assignment: dict[str, object],
        worker_output: dict[str, object],
        review_round: int,
    ) -> dict[str, object]:
        issues: list[str] = []
        output_text = str(worker_output.get("output", "")).lower()
        if not output_text or "fallback" in output_text:
            issues.append("Provide a stronger implementation summary for the assigned subtask.")
        failed_tools = [item for item in worker_output.get("tool_calls", []) if not bool(item.get("success"))]
        if failed_tools:
            issues.append("Resolve the failed tool request before approval.")
        if issues and review_round < MAX_REVIEW_ROUNDS:
            return {"status": "needs_fix", "comments": issues, "note": "Needs targeted fixes before approval."}
        return {"status": "approved", "comments": [], "note": "Approved."}

    def merge_worker_outputs(self, team_task: str, approved_outputs: dict[str, dict[str, object]]) -> str:
        output_lines = "\n".join(f"{item['worker_name']}: {item['output']}" for item in approved_outputs.values())
        return invoke_prompt(
            prompt=merge_prompt(self.spec),
            variables={"team_task": team_task, "approved_outputs": output_lines},
            role="lead",
            model=self.spec.model,
        )

    def build_default_subtasks(self, project_request: str, execution_mode: str, team_task: str) -> list[dict[str, object]]:
        app_name = infer_app_name(project_request)
        if execution_mode == "PROJECT_QUERY":
            return [
                {
                    "subtask": f"Explain the {self.spec.team} architecture for {app_name}.",
                    "expected_outcome": "Repository explanation with relevant files and design summary.",
                    "complexity": "complex",
                }
            ]
        if self.spec.team == "frontend":
            return [
                {
                    "subtask": "Design client-side state and interaction logic.",
                    "expected_outcome": "Clear frontend interaction behavior for the requested experience.",
                    "complexity": "complex",
                },
                {
                    "subtask": "Implement primary UI structure and user flows.",
                    "expected_outcome": "Functional interface covering the main user path.",
                    "complexity": "complex",
                },
                {
                    "subtask": "Implement responsive styling and visual polish.",
                    "expected_outcome": "Readable responsive interface with consistent styling.",
                    "complexity": "simple",
                },
                {
                    "subtask": "Document frontend behavior and usage notes.",
                    "expected_outcome": "Short usage-oriented handoff notes for the frontend changes.",
                    "complexity": "simple",
                },
            ]
        if self.spec.team == "backend":
            return [
                {
                    "subtask": "Design service boundaries and business logic behavior.",
                    "expected_outcome": "Clear backend structure for the requested behavior.",
                    "complexity": "complex",
                },
                {
                    "subtask": "Define API or schema behavior for integrations.",
                    "expected_outcome": "Stable contract for backend interactions.",
                    "complexity": "complex",
                },
                {
                    "subtask": "Implement the primary backend entry behavior.",
                    "expected_outcome": "Runnable entrypoint or request handling path.",
                    "complexity": "simple",
                },
            ]
        if self.spec.team == "qa":
            return [
                {
                    "subtask": "Define risk-focused validation strategy.",
                    "expected_outcome": "Coverage plan for critical user behavior.",
                    "complexity": "complex",
                },
                {
                    "subtask": "Create smoke and regression validation notes.",
                    "expected_outcome": "Short actionable QA checklist.",
                    "complexity": "simple",
                },
            ]
        return [
            {
                "subtask": "Prepare deployment and runtime workflow support.",
                "expected_outcome": "Operational support for build and release workflow.",
                "complexity": "complex",
            },
            {
                "subtask": "Document CI or environment setup implications.",
                "expected_outcome": "Clear repository workflow notes for operations.",
                "complexity": "simple",
            },
        ]


class WorkerAgent:
    """Worker agent responsible for a local execution summary."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def run(
        self,
        project_request: str,
        execution_mode: str,
        active_project: str,
        team_task: str,
        subtask: str,
        expected_outcome: str,
        review_comments: list[str] | None = None,
    ) -> WorkerDelivery:
        parser = PydanticOutputParser(pydantic_object=WorkerDelivery)
        messages = build_message_history(
            prompt=worker_execution_prompt(self.spec),
            variables={
                "execution_mode": execution_mode,
                "active_project": active_project,
                "project_request": project_request,
                "team_task": team_task,
                "subtask": subtask,
                "expected_outcome": expected_outcome,
                "format_instructions": parser.get_format_instructions(),
            },
            follow_up_human_input=(
                "Lead review comments:\n" + "\n".join(f"- {item}" for item in review_comments)
                if review_comments
                else None
            ),
        )
        try:
            return invoke_messages_structured(messages=messages, parser=parser, role="worker", model=self.spec.model)
        except Exception:
            return WorkerDelivery(
                summary=f"Execution fallback: {subtask}",
                commit_message=f"Implement {subtask.lower()}",
                implementation_notes=[],
                risks=[],
            )

    def execute(
        self,
        project_request: str,
        execution_mode: str,
        active_project: str,
        team_task: str,
        assignment: dict[str, object],
        file_owner: dict[str, str],
        created_files: dict[str, str],
        approval_manager: ApprovalManager | None = None,
        review_comments: list[str] | None = None,
    ) -> dict[str, object]:
        planned_files: list[str] = []
        subtask = str(assignment["subtask"])
        context = ToolExecutionContext(
            agent_name=self.spec.name,
            role=self.spec.role,
            team=self.spec.team,
            seniority=self.spec.seniority,
            model=(self.spec.model or ""),
        )
        allowed_tools = sorted(get_allowed_tools(context))
        tool_requests = infer_tool_requests(
            project_request=project_request,
            team_task=team_task,
            subtask=subtask,
            allowed_tools=allowed_tools,
            team=self.spec.team,
            active_project=active_project,
            request_type=execution_mode,
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
        delivery = self.run(
            project_request=project_request,
            execution_mode=execution_mode,
            active_project=active_project,
            team_task=team_task,
            subtask=subtask,
            expected_outcome=str(assignment.get("expected_outcome", "")),
            review_comments=review_comments,
        )
        output = delivery.summary
        tool_summary = build_tool_summary(tool_results)
        if tool_summary:
            output = f"{output}\nTool usage: {tool_summary}"
        return {
            "team": self.spec.team,
            "worker_key": self.spec.key,
            "worker_name": self.spec.name,
            "seniority": self.spec.seniority,
            "subtask": subtask,
            "output": output,
            "commit_message": delivery.commit_message,
            "implementation_notes": delivery.implementation_notes,
            "risks": delivery.risks,
            "tool_calls": tool_results,
            "created_files": collect_created_files(tool_results, self.spec.name),
            "created_paths": collect_created_paths(tool_results),
        }


class IntegrationAgent:
    """Final integration agent for cross-team synthesis."""

    def run(self, project_request: str, execution_mode: str, active_project: str, merged_output: str) -> str:
        return invoke_prompt(
            prompt=integration_prompt(),
            variables={
                "execution_mode": execution_mode,
                "active_project": active_project,
                "project_request": project_request,
                "merged_output": merged_output,
            },
            role="integration",
        )


def build_tool_summary(tool_results: list[dict[str, object]]) -> str:
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


def collect_created_files(tool_results: list[dict[str, object]], owner_name: str) -> dict[str, str]:
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
    created: list[str] = []
    for item in tool_results:
        if not bool(item.get("success")):
            continue
        for path in item.get("touched_paths", []):
            normalized = str(path)
            if normalized not in created:
                created.append(normalized)
    return created


def normalize_review_decisions(decisions: dict[str, dict[str, object]], review_round: int) -> dict[str, dict[str, object]]:
    if review_round < MAX_REVIEW_ROUNDS:
        return decisions
    normalized: dict[str, dict[str, object]] = {}
    for worker_key, decision in decisions.items():
        if str(decision.get("status")) == "needs_fix":
            normalized[worker_key] = {
                "status": "approved",
                "comments": list(decision.get("comments", [])),
                "note": "Approved after final review round with follow-up notes recorded.",
            }
        else:
            normalized[worker_key] = decision
    return normalized
