"""Intent routing agent implementation."""

from __future__ import annotations

from core.llm import invoke_structured
from core.prompts import intent_router_prompt
from langchain_core.output_parsers import PydanticOutputParser
from schemas.llm_outputs import IntentClassification


class IntentRouterAgent:
    """Classify whether a request is generic chat or project-related."""

    def run(self, project_request: str) -> dict[str, str]:
        """Return a structured request classification."""
        parser = PydanticOutputParser(pydantic_object=IntentClassification)
        try:
            result = invoke_structured(
                prompt=intent_router_prompt(),
                variables={
                    "project_request": project_request,
                    "format_instructions": parser.get_format_instructions(),
                },
                parser=parser,
                role="router",
            )
            return result.model_dump()
        except Exception:
            lowered = project_request.lower()
            project_tokens = {
                "build",
                "create",
                "modify",
                "fix",
                "repo",
                "repository",
                "architecture",
                "codebase",
                "project",
                "feature",
                "bug",
                "auth",
                "api",
                "frontend",
                "backend",
            }
            request_type = "PROJECT_RELATED" if any(token in lowered for token in project_tokens) else "GENERIC_CHAT"
            return {
                "request_type": request_type,
                "rationale": "Fallback classification used because the structured router was unavailable.",
            }
