"""Developer-focused agent implementation."""

from __future__ import annotations

from core.llm import generate_response


class DevAgent:
    """A minimal agent that turns a task string into a model response."""

    def build_prompt(self, task: str) -> str:
        """Create a prompt for the development agent."""
        return (
            "You are a helpful software development agent.\n"
            "Read the task and provide a concise, practical response.\n\n"
            f"Task: {task}"
        )

    def run(self, task: str) -> str:
        """Generate the agent response for a single task."""
        return generate_response(self.build_prompt(task))
