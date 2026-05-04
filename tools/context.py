"""Execution context models for tool usage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolExecutionContext:
    """Context carried with every tool request."""

    agent_name: str
    role: str
    team: str
    seniority: str
    model: str = ""

    @classmethod
    def system(cls) -> "ToolExecutionContext":
        """Return a privileged system context for internal server use."""
        return cls(
            agent_name="System Router",
            role="system",
            team="platform",
            seniority="system",
            model="",
        )
