"""File ownership checks for worker-scoped tool usage."""

from __future__ import annotations

from pathlib import PurePosixPath

from tools.context import ToolExecutionContext

MUTATING_FILE_TOOLS = {
    "write_file",
    "append_file",
    "edit_file",
    "delete_file",
    "create_folder",
}


def normalize_workspace_path(path: str) -> str:
    """Normalize a workspace-relative path for ownership checks."""
    normalized = str(PurePosixPath(path.replace("\\", "/"))).lstrip("./")
    return normalized


def validate_file_access(
    tool_name: str,
    params: dict[str, object],
    context: ToolExecutionContext,
    assigned_files: tuple[str, ...] = (),
    file_owner: dict[str, str] | None = None,
    created_files: dict[str, str] | None = None,
) -> str | None:
    """Return an error string when a worker attempts to touch an unowned path."""
    if context.role != "worker" or tool_name not in MUTATING_FILE_TOOLS:
        return None

    file_owner = file_owner or {}
    created_files = created_files or {}
    normalized_assigned = tuple(normalize_workspace_path(path) for path in assigned_files)
    target = normalize_workspace_path(str(params.get("path", "")))

    if not target:
        return f"{tool_name} requires a path."

    if tool_name == "create_folder":
        if any(path == target or path.startswith(f"{target}/") for path in normalized_assigned):
            return None
        return f"{context.agent_name} cannot create unrelated folder {target}."

    if target not in normalized_assigned:
        owner = file_owner.get(target)
        if owner and owner != context.agent_name:
            return f"{target} is assigned to {owner}, not {context.agent_name}."
        return f"{context.agent_name} is not assigned to modify {target}."

    owner = file_owner.get(target)
    if owner and owner != context.agent_name:
        return f"{target} is assigned to {owner}, not {context.agent_name}."

    creator = created_files.get(target)
    if creator and creator != context.agent_name and tool_name == "write_file":
        return f"{target} was already created by {creator}."

    return None
