"""GitHub-first repository workflow helpers."""

from __future__ import annotations

from typing import Any

from tools.approvals import ApprovalManager
from tools.context import ToolExecutionContext
from tools.git_tools import get_current_branch
from tools.git_tools import git_branch_exists
from tools.git_tools import git_has_remote
from tools.tool_router import execute_tool


class GitHubOpsAgent:
    """Prepare repository state and finalize approved branch merges."""

    def __init__(self, approval_manager: ApprovalManager | None = None) -> None:
        self.approval_manager = approval_manager
        self.context = ToolExecutionContext.system()

    def prepare_repository(
        self,
        active_project: str,
        repo_plan: dict[str, object],
    ) -> dict[str, object]:
        """Prepare main and optional project branch before delivery work starts."""
        repo_events: list[dict[str, object]] = []
        branches_created: list[str] = []
        has_remote = git_has_remote()
        current_branch = get_current_branch()

        if not git_branch_exists("main"):
            repo_events.append(self._run_tool("git_create_branch", {"branch": "main", "from_branch": current_branch}))
            branches_created.append("main")
        repo_events.append(self._run_tool("git_checkout", {"branch": "main"}))
        if bool(repo_plan.get("sync_with_remote")) and has_remote:
            repo_events.append(self._run_tool("git_pull", {"remote": "origin", "branch": "main"}))

        project_branch = str(repo_plan.get("project_branch", "")).strip()
        if project_branch:
            if not git_branch_exists(project_branch):
                repo_events.append(
                    self._run_tool(
                        "git_create_branch",
                        {"branch": project_branch, "from_branch": "main"},
                    )
                )
                branches_created.append(project_branch)
            repo_events.append(self._run_tool("git_checkout", {"branch": "main"}))

        return {
            "current_branch": current_branch,
            "main_branch_ready": True,
            "project_branch": project_branch,
            "active_project": active_project,
            "has_remote": has_remote,
            "notes": str(repo_plan.get("notes", "")),
            "repo_events": repo_events,
            "branches_created": branches_created,
        }

    def merge_approved_branches(
        self,
        approved_outputs: dict[str, dict[str, object]],
    ) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
        """Merge approved worker branches into main sequentially."""
        merge_status: dict[str, dict[str, object]] = {}
        repo_events: list[dict[str, object]] = []
        repo_events.append(self._run_tool("git_checkout", {"branch": "main"}))
        if git_has_remote():
            repo_events.append(self._run_tool("git_pull", {"remote": "origin", "branch": "main"}))

        merged_any = False
        for worker_key, output in approved_outputs.items():
            branch_name = str(output.get("branch_name", "")).strip()
            if not branch_name:
                merge_status[worker_key] = {
                    "status": "skipped",
                    "branch_name": "",
                    "message": "No branch was submitted for this worker.",
                }
                continue
            merge_event = self._run_tool("git_merge", {"branch": branch_name})
            repo_events.append(merge_event)
            if bool(merge_event.get("success")):
                merge_status[worker_key] = {
                    "status": "merged",
                    "branch_name": branch_name,
                    "message": "Merged to main.",
                }
                merged_any = True
            else:
                merge_status[worker_key] = {
                    "status": "merge_failed",
                    "branch_name": branch_name,
                    "message": str(merge_event.get("error", "merge failed")),
                }

        if merged_any and git_has_remote():
            repo_events.append(self._run_tool("git_push", {"remote": "origin", "branch": "main"}))

        return merge_status, repo_events

    def _run_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run one repository action through the audited tool router."""
        return execute_tool(
            tool_name=tool_name,
            params=params,
            context=self.context,
            approval_manager=self.approval_manager,
        )
