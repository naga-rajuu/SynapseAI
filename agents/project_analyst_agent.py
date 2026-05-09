"""Repository analyst agent for project-query mode."""

from __future__ import annotations

from core.llm import invoke_structured
from core.prompts import project_analyst_prompt
from core.repository import build_repo_snapshot
from langchain_core.output_parsers import PydanticOutputParser
from schemas.llm_outputs import AnalystAnswer


class ProjectAnalystAgent:
    """Answer repository-specific questions without modifying code."""

    def run(self, project_request: str) -> dict[str, object]:
        """Return a structured repository explanation."""
        parser = PydanticOutputParser(pydantic_object=AnalystAnswer)
        snapshot = build_repo_snapshot(project_request)
        try:
            result = invoke_structured(
                prompt=project_analyst_prompt(),
                variables={
                    "project_request": project_request,
                    "repo_snapshot": snapshot,
                    "format_instructions": parser.get_format_instructions(),
                },
                parser=parser,
                role="analyst",
            )
            return result.model_dump()
        except Exception:
            return {
                "answer": (
                    "The repository uses a LangGraph orchestration flow centered on manager, lead, worker, "
                    "review, and final integration nodes. Review orchestrator/graph.py, agents/, core/, and tools/."
                ),
                "relevant_files": [
                    "orchestrator/graph.py",
                    "agents/manager_agent.py",
                    "agents/dev_agent.py",
                    "core/llm.py",
                    "core/prompts.py",
                ],
                "components": ["LangGraph workflow", "LLM wrapper", "Agent hierarchy", "Tool router"],
                "dependencies": ["langgraph", "langchain", "langchain-core", "fastapi", "pydantic"],
            }
