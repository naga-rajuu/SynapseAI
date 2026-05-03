"""Engineering lead, worker, and integration agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from agents.agent_factory import AgentSpec
from core.llm import generate_response
from core.llm import is_error_response


class TeamLeadAgent:
    """Lead agent responsible for breaking down team work and merging outputs."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def build_worker_plan_prompt(self, project_request: str, team_task: str, workers: Iterable[AgentSpec]) -> str:
        worker_lines = "\n".join(
            f"{worker.name}: {worker.seniority} - {worker.focus}" for worker in workers
        )
        return (
            f"You are {self.spec.name}.\n"
            f"Your focus is {self.spec.focus}.\n"
            "Split the team task into one concise subtask per worker.\n"
            "Return exactly one line per worker in this format: Worker Name: subtask.\n"
            "No intro and no extra commentary.\n\n"
            f"Project request: {project_request}\n"
            f"Team task: {team_task}\n"
            f"Workers:\n{worker_lines}"
        )

    def plan_worker_tasks(
        self,
        project_request: str,
        team_task: str,
        workers: list[AgentSpec],
    ) -> list[dict[str, str]]:
        """Create one subtask per worker."""
        raw_response = generate_response(
            self.build_worker_plan_prompt(project_request, team_task, workers)
        )
        if is_error_response(raw_response):
            return [
                {
                    "worker": worker.name,
                    "seniority": worker.seniority,
                    "subtask": f"{worker.focus} for {team_task}",
                }
                for worker in workers
            ]

        parsed_assignments: list[dict[str, str]] = []
        worker_names = {worker.name: worker for worker in workers}
        for line in raw_response.splitlines():
            if ":" not in line:
                continue
            worker_name, subtask = line.split(":", 1)
            normalized_name = worker_name.strip()
            if normalized_name in worker_names and subtask.strip():
                worker = worker_names[normalized_name]
                parsed_assignments.append(
                    {
                        "worker": worker.name,
                        "seniority": worker.seniority,
                        "subtask": subtask.strip(),
                    }
                )

        if len(parsed_assignments) == len(workers):
            return parsed_assignments

        return [
            {
                "worker": worker.name,
                "seniority": worker.seniority,
                "subtask": f"{worker.focus} for {team_task}",
            }
            for worker in workers
        ]

    def build_merge_prompt(self, team_task: str, worker_outputs: list[dict[str, str]]) -> str:
        output_lines = "\n".join(
            f"{item['worker']}: {item['output']}" for item in worker_outputs
        )
        return (
            f"You are {self.spec.name}.\n"
            "Merge worker outputs into a concise team delivery summary.\n"
            "Return 3 to 5 short lines. No intro.\n\n"
            f"Team task: {team_task}\n"
            f"Worker outputs:\n{output_lines}"
        )

    def merge_worker_outputs(self, team_task: str, worker_outputs: list[dict[str, str]]) -> str:
        """Merge team worker outputs into one lead summary."""
        merged = generate_response(self.build_merge_prompt(team_task, worker_outputs))
        if is_error_response(merged):
            return "\n".join(
                f"- {item['worker']}: {item['output']}" for item in worker_outputs
            )
        return merged


class WorkerAgent:
    """Worker agent responsible for a local execution summary."""

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def build_prompt(self, project_request: str, team_task: str, subtask: str) -> str:
        role_focus = (
            "architecture, optimization, and review quality"
            if self.spec.seniority == "senior"
            else "implementation, boilerplate, and practical execution"
        )
        return (
            f"You are {self.spec.name}.\n"
            f"Team: {self.spec.team}.\n"
            f"Focus on {role_focus}.\n"
            "Respond with a concise execution summary for only your subtask.\n"
            "Keep it under 5 lines and avoid repeating the full project context.\n\n"
            f"Project request: {project_request}\n"
            f"Team task: {team_task}\n"
            f"Your subtask: {subtask}"
        )

    def run(self, project_request: str, team_task: str, subtask: str) -> str:
        """Return the worker output for a local subtask."""
        output = generate_response(self.build_prompt(project_request, team_task, subtask))
        if is_error_response(output):
            return f"Execution fallback: {subtask}"
        return output


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


def run_parallel_workers(
    project_request: str,
    team_task: str,
    worker_specs: list[AgentSpec],
    assignments: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Execute worker subtasks in parallel for a single lead."""

    def run_single_worker(worker_spec: AgentSpec, subtask: str) -> dict[str, str]:
        agent = WorkerAgent(worker_spec)
        output = agent.run(project_request, team_task, subtask)
        return {
            "team": worker_spec.team,
            "worker": worker_spec.name,
            "seniority": worker_spec.seniority,
            "subtask": subtask,
            "output": output,
        }

    assignment_map = {item["worker"]: item["subtask"] for item in assignments}
    with ThreadPoolExecutor(max_workers=len(worker_specs)) as executor:
        futures = [
            executor.submit(run_single_worker, worker_spec, assignment_map[worker_spec.name])
            for worker_spec in worker_specs
        ]
        return [future.result() for future in futures]
