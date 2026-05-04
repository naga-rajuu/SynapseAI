"""Human-in-the-loop approval helpers."""

from __future__ import annotations

from threading import Lock
from typing import Protocol

from tools.context import ToolExecutionContext


class ApprovalManager(Protocol):
    """Protocol for approval managers."""

    def request_approval(
        self,
        context: ToolExecutionContext,
        action_label: str,
        reason: str,
    ) -> bool:
        """Return True when the tool request is approved."""


class TerminalApprovalManager:
    """Serialize terminal prompts for risky tool actions."""

    _lock = Lock()

    def request_approval(
        self,
        context: ToolExecutionContext,
        action_label: str,
        reason: str,
    ) -> bool:
        with self._lock:
            print("Approval Needed:")
            print(f"Agent: {context.agent_name}")
            print(f"Action: {action_label}")
            print(f"Reason: {reason}")
            decision = input("Approve? (y/n): ").strip().lower()
            return decision == "y"


class PreapprovedApprovalManager:
    """Approval manager for API requests or tests."""

    def __init__(self, approved: bool | None = None) -> None:
        self.approved = approved

    def request_approval(
        self,
        context: ToolExecutionContext,
        action_label: str,
        reason: str,
    ) -> bool:
        return self.approved is True
