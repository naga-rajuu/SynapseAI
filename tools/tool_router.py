"""Tool routing, permissions, approvals, and audit handling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tools.approvals import ApprovalManager
from tools.approvals import PreapprovedApprovalManager
from tools.audit import get_audit_log_path
from tools.audit import summarize_output
from tools.audit import write_audit_entry
from tools.context import ToolExecutionContext
from tools.file_guard import normalize_workspace_path
from tools.file_guard import validate_file_access
from tools.file_tools import append_file
from tools.file_tools import create_folder
from tools.file_tools import delete_file
from tools.file_tools import edit_file
from tools.file_tools import list_files
from tools.file_tools import read_file
from tools.file_tools import write_file
from tools.git_tools import git_add
from tools.git_tools import git_branch
from tools.git_tools import git_checkout
from tools.git_tools import git_commit
from tools.git_tools import git_diff
from tools.git_tools import git_log
from tools.git_tools import git_push
from tools.git_tools import git_status
from tools.permissions import get_allowed_tools
from tools.permissions import requires_approval
from tools.search_tools import dependency_lookup
from tools.search_tools import find_file
from tools.search_tools import grep_keyword
from tools.search_tools import search_code
from tools.shell_tools import build_project
from tools.shell_tools import install_packages
from tools.shell_tools import run_command
from tools.shell_tools import run_pytest
from tools.shell_tools import run_tests
from tools.shell_tools import start_server
from tools.shell_tools import stop_process

ToolHandler = Callable[[dict[str, Any]], Any]


def _handle_write_file(params: dict[str, Any]) -> Any:
    return write_file(path=str(params["path"]), content=str(params["content"]))


def _handle_read_file(params: dict[str, Any]) -> Any:
    return read_file(path=str(params["path"]))


def _handle_edit_file(params: dict[str, Any]) -> Any:
    return edit_file(
        path=str(params["path"]),
        old_text=str(params["old_text"]),
        new_text=str(params["new_text"]),
    )


def _handle_append_file(params: dict[str, Any]) -> Any:
    return append_file(path=str(params["path"]), content=str(params["content"]))


def _handle_create_folder(params: dict[str, Any]) -> Any:
    return create_folder(path=str(params["path"]))


def _handle_delete_file(params: dict[str, Any]) -> Any:
    return delete_file(path=str(params["path"]))


def _handle_list_files(params: dict[str, Any]) -> Any:
    return list_files(directory=str(params.get("directory", ".")))


def _handle_find_file(params: dict[str, Any]) -> Any:
    return find_file(name=str(params["name"]), directory=str(params.get("directory", ".")))


def _handle_grep_keyword(params: dict[str, Any]) -> Any:
    return grep_keyword(keyword=str(params["keyword"]), directory=str(params.get("directory", ".")))


def _handle_search_code(params: dict[str, Any]) -> Any:
    return search_code(query=str(params["query"]), directory=str(params.get("directory", ".")))


def _handle_dependency_lookup(params: dict[str, Any]) -> Any:
    return dependency_lookup(name=str(params["name"]))


def _handle_git_status(params: dict[str, Any]) -> Any:
    return git_status()


def _handle_git_branch(params: dict[str, Any]) -> Any:
    return git_branch()


def _handle_git_checkout(params: dict[str, Any]) -> Any:
    return git_checkout(branch=str(params["branch"]))


def _handle_git_add(params: dict[str, Any]) -> Any:
    paths = params.get("paths")
    return git_add(paths=list(paths) if isinstance(paths, list) else None)


def _handle_git_commit(params: dict[str, Any]) -> Any:
    return git_commit(message=str(params["message"]))


def _handle_git_diff(params: dict[str, Any]) -> Any:
    target = params.get("target")
    return git_diff(target=str(target) if target else None)


def _handle_git_push(params: dict[str, Any]) -> Any:
    return git_push(
        remote=str(params.get("remote", "origin")),
        branch=str(params.get("branch", "main")),
    )


def _handle_git_log(params: dict[str, Any]) -> Any:
    return git_log(limit=int(params.get("limit", 10)))


def _handle_run_tests(params: dict[str, Any]) -> Any:
    return run_tests()


def _handle_run_pytest(params: dict[str, Any]) -> Any:
    return run_pytest(target=str(params.get("target", ".")))


def _handle_install_packages(params: dict[str, Any]) -> Any:
    packages = params.get("packages")
    if not isinstance(packages, list):
        raise ValueError("packages must be a list.")
    return install_packages(packages=[str(item) for item in packages])


def _handle_build_project(params: dict[str, Any]) -> Any:
    return build_project()


def _handle_start_server(params: dict[str, Any]) -> Any:
    command = params.get("command")
    if not isinstance(command, list):
        raise ValueError("command must be a list.")
    return start_server(name=str(params["name"]), command=[str(item) for item in command])


def _handle_stop_process(params: dict[str, Any]) -> Any:
    return stop_process(name=str(params["name"]))


def _handle_run_command(params: dict[str, Any]) -> Any:
    command = params.get("command")
    if not isinstance(command, list):
        raise ValueError("command must be a list.")
    return run_command([str(item) for item in command])


TOOL_REGISTRY: dict[str, ToolHandler] = {
    "write_file": _handle_write_file,
    "read_file": _handle_read_file,
    "edit_file": _handle_edit_file,
    "append_file": _handle_append_file,
    "create_folder": _handle_create_folder,
    "delete_file": _handle_delete_file,
    "list_files": _handle_list_files,
    "find_file": _handle_find_file,
    "grep_keyword": _handle_grep_keyword,
    "search_code": _handle_search_code,
    "dependency_lookup": _handle_dependency_lookup,
    "git_status": _handle_git_status,
    "git_branch": _handle_git_branch,
    "git_checkout": _handle_git_checkout,
    "git_add": _handle_git_add,
    "git_commit": _handle_git_commit,
    "git_diff": _handle_git_diff,
    "git_push": _handle_git_push,
    "git_log": _handle_git_log,
    "run_tests": _handle_run_tests,
    "run_pytest": _handle_run_pytest,
    "install_packages": _handle_install_packages,
    "build_project": _handle_build_project,
    "start_server": _handle_start_server,
    "stop_process": _handle_stop_process,
    "run_command": _handle_run_command,
}


def execute_tool(
    tool_name: str,
    params: dict[str, Any],
    context: ToolExecutionContext | None = None,
    approval_manager: ApprovalManager | None = None,
    assigned_files: tuple[str, ...] = (),
    file_owner: dict[str, str] | None = None,
    created_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a supported tool with permissions, approvals, and audit logging."""
    context = context or ToolExecutionContext.system()
    approval_manager = approval_manager or PreapprovedApprovalManager(approved=True)
    action_label = f"{tool_name}({summarize_output(params)})"

    if tool_name not in TOOL_REGISTRY:
        result = build_result(
            success=False,
            tool=tool_name,
            result=None,
            error=f"Invalid tool name: {tool_name}",
            approval_required=False,
            touched_paths=extract_touched_paths(tool_name, params),
        )
        log_tool_call(context, tool_name, params, approved=False, success=False, output=result["error"])
        return result

    allowed_tools = get_allowed_tools(context)
    if tool_name not in allowed_tools:
        error = f"{context.agent_name} is not allowed to use {tool_name}."
        result = build_result(
            success=False,
            tool=tool_name,
            result=None,
            error=error,
            approval_required=False,
            touched_paths=extract_touched_paths(tool_name, params),
        )
        log_tool_call(context, tool_name, params, approved=False, success=False, output=error)
        return result

    file_access_error = validate_file_access(
        tool_name=tool_name,
        params=params,
        context=context,
        assigned_files=assigned_files,
        file_owner=file_owner,
        created_files=created_files,
    )
    if file_access_error:
        result = build_result(
            success=False,
            tool=tool_name,
            result=None,
            error=file_access_error,
            approval_required=False,
            touched_paths=extract_touched_paths(tool_name, params),
        )
        log_tool_call(context, tool_name, params, approved=False, success=False, output=file_access_error)
        return result

    approval_reason = requires_approval(tool_name, params)
    approved = True
    if approval_reason:
        approved = approval_manager.request_approval(context, action_label, approval_reason)
        if not approved:
            error = f"Approval rejected for {tool_name}."
            result = build_result(
                success=False,
                tool=tool_name,
                result=None,
                error=error,
                approval_required=True,
                touched_paths=extract_touched_paths(tool_name, params),
            )
            log_tool_call(context, tool_name, params, approved=False, success=False, output=error)
            return result

    try:
        tool_result = TOOL_REGISTRY[tool_name](params)
    except (KeyError, ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        error = str(exc)
        result = build_result(
            success=False,
            tool=tool_name,
            result=None,
            error=error,
            approval_required=False,
            touched_paths=extract_touched_paths(tool_name, params),
        )
        log_tool_call(context, tool_name, params, approved=approved, success=False, output=error)
        return result

    result = build_result(
        success=True,
        tool=tool_name,
        result=tool_result,
        error=None,
        approval_required=False,
        touched_paths=extract_touched_paths(tool_name, params),
    )
    log_tool_call(context, tool_name, params, approved=approved, success=True, output=tool_result)
    return result


def build_result(
    success: bool,
    tool: str,
    result: Any,
    error: str | None,
    approval_required: bool,
    touched_paths: list[str],
) -> dict[str, Any]:
    """Normalize tool execution responses."""
    return {
        "success": success,
        "tool": tool,
        "result": result,
        "error": error,
        "approval_required": approval_required,
        "audit_log_path": get_audit_log_path(),
        "touched_paths": touched_paths,
    }


def extract_touched_paths(tool_name: str, params: dict[str, Any]) -> list[str]:
    """Return the normalized workspace paths affected by a tool call."""
    if tool_name in {
        "write_file",
        "read_file",
        "edit_file",
        "append_file",
        "create_folder",
        "delete_file",
    }:
        path = params.get("path")
        if path:
            return [normalize_workspace_path(str(path))]
    return []


def log_tool_call(
    context: ToolExecutionContext,
    tool_name: str,
    params: dict[str, Any],
    approved: bool,
    success: bool,
    output: Any,
) -> None:
    """Write a tool execution event to the audit log."""
    write_audit_entry(
        {
            "agent_name": context.agent_name,
            "role": context.role,
            "team": context.team,
            "seniority": context.seniority,
            "tool_name": tool_name,
            "parameters": params,
            "approved": approved,
            "success": success,
            "output_summary": summarize_output(output),
        }
    )
