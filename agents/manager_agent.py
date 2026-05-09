"""Manager agent implementation."""

from __future__ import annotations

import re

from agents.agent_factory import get_team_names
from core.llm import invoke_prompt
from core.llm import invoke_structured
from core.prompts import manager_plan_prompt
from core.prompts import onboarding_manager_prompt
from core.repository import infer_active_project_name
from langchain_core.output_parsers import PydanticOutputParser
from schemas.llm_outputs import ManagerPlan
from schemas.llm_outputs import OnboardingQuestionSet

ONBOARDING_FIELDS = [
    "repo_mode",
    "github_username",
    "repo_name",
    "repo_visibility",
    "token_ready",
    "preferred_stack",
    "major_features",
]


class ManagerAgent:
    """Manager responsible for onboarding intake and execution planning."""

    def collect_onboarding_details(
        self,
        project_request: str,
        gathered_requirements: dict[str, str],
        missing_fields: list[str],
        validation_errors: list[str],
    ) -> dict[str, str]:
        """Ask only for missing onboarding details and merge the response."""
        known_details = format_known_details(gathered_requirements)
        parser = PydanticOutputParser(pydantic_object=OnboardingQuestionSet)
        try:
            prompt_text = invoke_structured(
                prompt=onboarding_manager_prompt(),
                variables={
                    "project_request": project_request,
                    "known_details": known_details,
                    "missing_fields": "\n".join(f"- {item}" for item in missing_fields) or "None",
                    "validation_errors": "\n".join(f"- {item}" for item in validation_errors) or "None",
                    "format_instructions": parser.get_format_instructions(),
                },
                parser=parser,
                role="manager",
            ).message
        except Exception:
            prompt_text = build_fallback_onboarding_prompt(missing_fields, validation_errors)

        print(prompt_text)
        response = input("> ").strip()
        extracted = extract_onboarding_fields(response)
        return {**gathered_requirements, **extracted}

    def plan_execution(
        self,
        project_request: str,
        gathered_requirements: dict[str, str],
        repo_context: dict[str, object],
    ) -> dict[str, object]:
        """Return a structured organization plan after onboarding passes."""
        parser = PydanticOutputParser(pydantic_object=ManagerPlan)
        try:
            result = invoke_structured(
                prompt=manager_plan_prompt(),
                variables={
                    "project_request": project_request,
                    "repo_context": format_known_details({key: str(value) for key, value in repo_context.items()}),
                    "requirements": format_known_details(gathered_requirements),
                    "team_roster": ", ".join(team.upper() for team in get_team_names()),
                    "format_instructions": parser.get_format_instructions(),
                },
                parser=parser,
                role="manager",
            )
            return {
                "execution_mode": result.execution_mode,
                "active_project": result.active_project,
                "execution_summary": result.execution_summary,
                "repo_plan": result.repo_plan.model_dump(),
                "team_tasks": {
                    "backend": result.backend.model_dump(),
                    "frontend": result.frontend.model_dump(),
                    "qa": result.qa.model_dump(),
                    "devops": result.devops.model_dump(),
                },
            }
        except Exception:
            return self.build_fallback_plan(project_request, gathered_requirements)

    def build_fallback_plan(
        self,
        project_request: str,
        gathered_requirements: dict[str, str],
    ) -> dict[str, object]:
        """Return a safe fallback when structured parsing fails."""
        lowered = project_request.lower()
        execution_mode = (
            "PROJECT_QUERY"
            if any(token in lowered for token in {"explain", "which files", "architecture", "how does"})
            else "MODIFY_PROJECT"
            if any(token in lowered for token in {"modify", "add", "change", "fix", "update"})
            else "BUILD_PROJECT"
        )
        active_project = gathered_requirements.get("repo_name") or infer_active_project_name(project_request)
        frontend_only = execution_mode == "MODIFY_PROJECT" and any(
            token in lowered for token in {"dark mode", "navbar", "theme", "layout", "ui", "frontend"}
        )
        return {
            "execution_mode": execution_mode,
            "active_project": active_project,
            "execution_summary": f"Execute {execution_mode.lower()} workflow for {project_request}.",
            "repo_plan": {
                "use_current_repo": True,
                "ensure_main_branch": True,
                "sync_with_remote": True,
                "project_branch": "",
                "notes": "Fallback planning path.",
            },
            "team_tasks": {
                "backend": {
                    "needed": execution_mode == "BUILD_PROJECT" or (execution_mode == "MODIFY_PROJECT" and not frontend_only),
                    "priority": "high" if execution_mode == "BUILD_PROJECT" else "medium",
                    "task": f"Implement backend work for {project_request}." if execution_mode != "PROJECT_QUERY" and not frontend_only else "",
                },
                "frontend": {
                    "needed": execution_mode != "PROJECT_QUERY",
                    "priority": "high",
                    "task": f"Implement frontend work for {project_request}." if execution_mode != "PROJECT_QUERY" else "",
                },
                "qa": {
                    "needed": execution_mode in {"BUILD_PROJECT", "MODIFY_PROJECT"},
                    "priority": "medium",
                    "task": f"Validate behavior for {project_request}." if execution_mode != "PROJECT_QUERY" else "",
                },
                "devops": {
                    "needed": execution_mode == "BUILD_PROJECT",
                    "priority": "low",
                    "task": f"Support repository and deployment workflow for {project_request}." if execution_mode == "BUILD_PROJECT" else "",
                },
            },
        }


def extract_onboarding_fields(text: str) -> dict[str, str]:
    """Extract onboarding fields from free-form user input."""
    extracted: dict[str, str] = {}
    lowered = text.lower()

    repo_mode_match = re.search(r"\b(existing|new)\b", lowered)
    if repo_mode_match:
        extracted["repo_mode"] = repo_mode_match.group(1)

    visibility_match = re.search(r"\b(public|private)\b", lowered)
    if visibility_match:
        extracted["repo_visibility"] = visibility_match.group(1)

    repo_name_match = re.search(r"(?:repo(?: name)?\s*[:=]\s*|repository\s*[:=]\s*)([A-Za-z0-9_.-]+)", text, re.IGNORECASE)
    if repo_name_match:
        extracted["repo_name"] = repo_name_match.group(1)

    username_match = re.search(r"(?:username|user|org|organization)\s*[:=]\s*([A-Za-z0-9_.-]+)", text, re.IGNORECASE)
    if username_match:
        extracted["github_username"] = username_match.group(1)

    if "token" in lowered:
        extracted["token_ready"] = "true" if any(token in lowered for token in {"ready", "configured", "yes", "available"}) else "false"

    stack_match = re.search(r"(?:stack|tech stack|preferred stack)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if stack_match:
        extracted["preferred_stack"] = stack_match.group(1).strip()

    features_match = re.search(r"(?:features|major features|requirements)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if features_match:
        extracted["major_features"] = features_match.group(1).strip()

    constraints_match = re.search(r"(?:constraints?)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if constraints_match:
        extracted["constraints"] = constraints_match.group(1).strip()

    branch_match = re.search(r"(?:branch policy|branch)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if branch_match:
        extracted["branch_policy"] = branch_match.group(1).strip()

    return extracted


def build_missing_fields(gathered_requirements: dict[str, str], project_request: str = "") -> list[str]:
    """Return the onboarding fields still required to proceed."""
    lowered = project_request.lower()
    is_query = any(token in lowered for token in {"explain", "architecture", "which files", "how does", "current project"})
    required_fields = ["repo_mode", "github_username", "repo_name", "repo_visibility"]
    if not (is_query and gathered_requirements.get("repo_mode") == "existing"):
        required_fields.append("token_ready")
    if not is_query:
        required_fields.extend(["preferred_stack", "major_features"])
    missing = [field for field in required_fields if not gathered_requirements.get(field)]
    if gathered_requirements.get("repo_mode") == "existing" and not gathered_requirements.get("branch_policy"):
        missing.append("branch_policy")
    return missing


def format_known_details(details: dict[str, str]) -> str:
    """Format known details for prompts."""
    if not details:
        return "None"
    return "\n".join(f"- {key}: {value}" for key, value in details.items())


def build_fallback_onboarding_prompt(missing_fields: list[str], validation_errors: list[str]) -> str:
    """Build a deterministic onboarding prompt when LLM prompting fails."""
    parts = ["Before engineering starts, I need the remaining GitHub and project details."]
    if missing_fields:
        parts.append("Missing fields: " + ", ".join(missing_fields))
    if validation_errors:
        parts.append("Validation issues: " + "; ".join(validation_errors))
    parts.append("Provide the missing values in one reply using key=value format where possible.")
    return "\n".join(parts)
