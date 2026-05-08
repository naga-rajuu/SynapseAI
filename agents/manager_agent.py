"""Manager agent implementation."""

from __future__ import annotations

from agents.agent_factory import get_team_names
from core.llm import invoke_structured
from core.llm import is_error_response
from core.prompts import manager_plan_prompt
from langchain_core.output_parsers import PydanticOutputParser
from schemas.llm_outputs import ManagerPlan


class ManagerAgent:
    """Agent responsible for turning a request into a short plan."""

    def run(self, project_request: str) -> dict[str, dict[str, object]] | str:
        parser = PydanticOutputParser(pydantic_object=ManagerPlan)
        try:
            result = invoke_structured(
                prompt=manager_plan_prompt(),
                variables={
                    "project_request": project_request,
                    "team_roster": ", ".join(team.upper() for team in get_team_names()),
                    "format_instructions": parser.get_format_instructions(),
                },
                parser=parser,
                role="manager",
            )
        except Exception as exc:  # pragma: no cover - graceful runtime fallback
            error = str(exc)
            if is_error_response(error):
                return error
            return self.build_fallback_plan(project_request)

        return {
            "backend": result.backend.model_dump(),
            "frontend": result.frontend.model_dump(),
            "qa": result.qa.model_dump(),
            "devops": result.devops.model_dump(),
        }

    def build_fallback_plan(self, project_request: str) -> dict[str, dict[str, object]]:
        """Return a safe fallback when structured parsing fails."""
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
