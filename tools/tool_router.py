"""Tool routing utilities."""

from __future__ import annotations

from tools.file_tools import list_files, read_file, write_file


def execute_tool(tool_name: str, params: dict) -> dict[str, object]:
    """Execute a supported tool with the provided params."""
    try:
        if tool_name == "write_file":
            result = write_file(
                path=str(params["path"]),
                content=str(params["content"]),
            )
        elif tool_name == "read_file":
            result = read_file(path=str(params["path"]))
        elif tool_name == "list_files":
            result = list_files(directory=str(params.get("directory", ".")))
        else:
            return {
                "success": False,
                "tool": tool_name,
                "result": None,
                "error": f"Invalid tool name: {tool_name}",
            }
    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        return {
            "success": False,
            "tool": tool_name,
            "result": None,
            "error": str(exc),
        }

    return {
        "success": True,
        "tool": tool_name,
        "result": result,
        "error": None,
    }
