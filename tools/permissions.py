"""Role-based tool permissions and approval rules."""

from __future__ import annotations

from pathlib import Path

from tools.context import ToolExecutionContext

READ_FILE_TOOLS = {
    "read_file",
    "list_files",
    "find_file",
    "grep_keyword",
    "search_code",
    "dependency_lookup",
}
WRITE_FILE_TOOLS = {"write_file", "append_file", "create_folder"}
EDIT_FILE_TOOLS = {"edit_file"}
DELETE_FILE_TOOLS = {"delete_file"}
SAFE_GIT_TOOLS = {"git_status", "git_branch", "git_diff", "git_log"}
RISKY_GIT_TOOLS = {"git_checkout", "git_create_branch", "git_add", "git_commit", "git_pull", "git_push", "git_merge"}
SAFE_SHELL_TOOLS = {"run_tests", "run_pytest", "build_project"}
RISKY_SHELL_TOOLS = {"install_packages", "start_server", "stop_process", "run_command"}

PROTECTED_BRANCHES = {"main", "master"}
SENSITIVE_PATH_SUFFIXES = {
    ".env",
    ".gitignore",
    "requirements.txt",
}
SENSITIVE_PATH_PARTS = {".git", ".venv", ".deps", "logs"}


def get_allowed_tools(context: ToolExecutionContext) -> set[str]:
    """Return the tool names allowed for a given agent context."""
    if context.role == "system":
        return (
            READ_FILE_TOOLS
            | WRITE_FILE_TOOLS
            | EDIT_FILE_TOOLS
            | DELETE_FILE_TOOLS
            | SAFE_GIT_TOOLS
            | RISKY_GIT_TOOLS
            | SAFE_SHELL_TOOLS
            | RISKY_SHELL_TOOLS
        )

    if context.role == "manager":
        return set()

    if context.role == "lead":
        return READ_FILE_TOOLS | WRITE_FILE_TOOLS | EDIT_FILE_TOOLS | SAFE_GIT_TOOLS | RISKY_GIT_TOOLS | SAFE_SHELL_TOOLS

    if context.role == "worker":
        return READ_FILE_TOOLS | WRITE_FILE_TOOLS | EDIT_FILE_TOOLS | SAFE_GIT_TOOLS | RISKY_GIT_TOOLS | SAFE_SHELL_TOOLS

    if context.team == "qa":
        return READ_FILE_TOOLS | SAFE_SHELL_TOOLS | SAFE_GIT_TOOLS | RISKY_GIT_TOOLS

    if context.seniority == "senior":
        return READ_FILE_TOOLS | WRITE_FILE_TOOLS | EDIT_FILE_TOOLS | SAFE_GIT_TOOLS | RISKY_GIT_TOOLS | SAFE_SHELL_TOOLS

    return READ_FILE_TOOLS | WRITE_FILE_TOOLS | SAFE_GIT_TOOLS | RISKY_GIT_TOOLS


def requires_approval(tool_name: str, params: dict[str, object]) -> str | None:
    """Return the approval reason for risky actions, otherwise None."""
    if tool_name in {"git_commit", "git_push", "git_merge"}:
        return f"{tool_name} changes repository history"

    if tool_name == "git_checkout":
        branch = str(params.get("branch", ""))
        if branch in PROTECTED_BRANCHES:
            return f"switching protected branch {branch}"

    if tool_name == "delete_file":
        return "delete_file removes workspace content"

    if tool_name in {"write_file", "append_file", "edit_file"}:
        path = str(params.get("path", ""))
        if is_sensitive_path(path):
            return f"{tool_name} targets a sensitive file"

    if tool_name == "install_packages":
        return "install_packages changes the runtime environment"

    if tool_name in {"start_server", "stop_process", "run_command"}:
        return f"{tool_name} manages or executes shell processes"

    return None


def is_sensitive_path(path: str) -> bool:
    """Return True for paths that should require approval before changes."""
    normalized = Path(path)
    if normalized.name in SENSITIVE_PATH_SUFFIXES:
        return True
    if any(part in SENSITIVE_PATH_PARTS for part in normalized.parts):
        return True
    return False
