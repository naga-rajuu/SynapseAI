"""Developer-focused agent implementation."""

from __future__ import annotations

from core.llm import generate_response


class DevAgent:
    """Agent responsible for executing a manager-produced plan."""

    def build_prompt(self, task_breakdown: str) -> str:
        """Create a prompt for the development agent."""
        return (
            "You are a developer agent.\n"
            "Execute the implementation plan and return a concise implementation response.\n"
            "Use only the plan provided below.\n"
            "Do not repeat the full plan.\n"
            "Keep the response short and practical.\n\n"
            f"Plan:\n{task_breakdown}"
        )

    def run(self, task_breakdown: str) -> str:
        """Generate the agent response for a single task breakdown."""
        return generate_response(self.build_prompt(task_breakdown))
