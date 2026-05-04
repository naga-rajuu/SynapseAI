"""Manager agent implementation."""

from __future__ import annotations

from agents.agent_factory import get_team_names
from core.llm import generate_response
from core.llm import is_error_response


class ManagerAgent:
    """Agent responsible for turning a request into a short plan."""

    def build_prompt(self, project_request: str) -> str:
        teams = ", ".join(team.upper() for team in get_team_names())
        return (
            "You are a manager agent.\n"
            "Decide which teams are needed for the project.\n"
            f"Return exactly one line for each team in this format: TEAM|YES|task or TEAM|NO|none.\n"
            f"Teams: {teams}.\n"
            "No intro, no bullets, and no extra commentary.\n\n"
            f"Project request: {project_request}"
        )

    def run(self, project_request: str) -> dict[str, dict[str, object]] | str:
        heuristic_tasks = self.infer_team_tasks_from_request(project_request)
        if heuristic_tasks is not None:
            return heuristic_tasks

        raw_response = generate_response(self.build_prompt(project_request))
        if is_error_response(raw_response):
            return raw_response
        return self.parse_team_tasks(raw_response, project_request)

    def parse_team_tasks(
        self,
        raw_response: str,
        project_request: str,
    ) -> dict[str, dict[str, object]]:
        """Parse team task lines from the LLM output."""
        parsed_tasks: dict[str, dict[str, object]] = {}
        for line in raw_response.splitlines():
            parts = [part.strip() for part in line.split("|", 2)]
            if len(parts) != 3:
                continue
            team_name, needed_flag, task = parts
            normalized_team = team_name.lower()
            if normalized_team in get_team_names():
                parsed_tasks[normalized_team] = {
                    "needed": needed_flag.upper() == "YES",
                    "task": "" if needed_flag.upper() != "YES" else task,
                }

        if len(parsed_tasks) == len(get_team_names()):
            return parsed_tasks

        return {
            "backend": {
                "needed": True,
                "task": f"Design backend services and APIs for {project_request}.",
            },
            "frontend": {
                "needed": True,
                "task": f"Design frontend flows and interfaces for {project_request}.",
            },
            "qa": {
                "needed": True,
                "task": f"Define test coverage and validation strategy for {project_request}.",
            },
            "devops": {
                "needed": True,
                "task": f"Prepare deployment, CI/CD, and runtime setup for {project_request}.",
            },
        }

    def infer_team_tasks_from_request(
        self,
        project_request: str,
    ) -> dict[str, dict[str, object]] | None:
        """Use simple intent heuristics when the request clearly targets one team."""
        normalized = project_request.lower()
        wants_frontend = any(token in normalized for token in {"ui", "frontend", "page", "screen", "component"})
        wants_backend = any(token in normalized for token in {"api", "backend", "database", "auth"})
        wants_qa = any(token in normalized for token in {"test", "qa", "bug", "verify"})
        wants_devops = any(token in normalized for token in {"docker", "deploy", "devops", "ci", "pipeline"})
        only_frontend = "only ui" in normalized or "only frontend" in normalized
        only_backend = "only backend" in normalized or "only api" in normalized

        if only_frontend or (wants_frontend and not wants_backend and not wants_qa and not wants_devops):
            return {
                "backend": {"needed": False, "task": ""},
                "frontend": {"needed": True, "task": "frontend development"},
                "qa": {"needed": False, "task": ""},
                "devops": {"needed": False, "task": ""},
            }

        if only_backend or (wants_backend and not wants_frontend and not wants_qa and not wants_devops):
            return {
                "backend": {"needed": True, "task": "backend implementation"},
                "frontend": {"needed": False, "task": ""},
                "qa": {"needed": False, "task": ""},
                "devops": {"needed": False, "task": ""},
            }

        return None
