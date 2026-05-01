"""Manager agent implementation."""

from __future__ import annotations

from core.llm import generate_response


class ManagerAgent:
    """Agent responsible for turning a request into a short plan."""

    def build_prompt(self, user_input: str) -> str:
        return (
            "You are a manager agent.\n"
            "Create a short implementation plan with 2 to 4 numbered steps.\n"
            "Return only the numbered steps.\n"
            "No intro, no headings, and no estimates.\n\n"
            f"Request: {user_input}"
        )

    def run(self, user_input: str) -> str:
        return generate_response(self.build_prompt(user_input))
