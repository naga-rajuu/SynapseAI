"""Repository inspection helpers for query and workflow planning."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SNAPSHOT_CANDIDATES = [
    "main.py",
    "orchestrator/graph.py",
    "agents/agent_factory.py",
    "agents/manager_agent.py",
    "agents/dev_agent.py",
    "core/llm.py",
    "core/prompts.py",
    "schemas/state.py",
    "schemas/llm_outputs.py",
    "tools/tool_router.py",
    "tools/permissions.py",
    "tools/git_tools.py",
    "requirements.txt",
]


def list_repo_files(limit: int = 120) -> list[str]:
    """Return a bounded repository file listing."""
    files: list[str] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if path.is_dir():
            continue
        if any(part in {".git", ".venv", "__pycache__", ".deps"} for part in path.parts):
            continue
        files.append(path.relative_to(PROJECT_ROOT).as_posix())
        if len(files) >= limit:
            break
    return files


def read_file_excerpt(relative_path: str, max_chars: int = 1800) -> str:
    """Read a truncated file excerpt for prompt context."""
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""
    return text[:max_chars].strip()


def build_repo_snapshot(question: str = "") -> str:
    """Build a compact repository snapshot for analyst/query mode."""
    files = list_repo_files()
    sections = ["Repository files:", *[f"- {item}" for item in files[:60]]]
    if question:
        lowered = question.lower()
        focus_files = [path for path in SNAPSHOT_CANDIDATES if any(token in path.lower() for token in lowered.split())]
    else:
        focus_files = []
    selected = list(dict.fromkeys(focus_files + SNAPSHOT_CANDIDATES[:8]))
    for relative_path in selected[:10]:
        excerpt = read_file_excerpt(relative_path)
        if excerpt:
            sections.append(f"\nFile: {relative_path}\n{excerpt}")
    return "\n".join(sections)


def infer_active_project_name(project_request: str) -> str:
    """Create a compact project name for workflow state."""
    words = [item for item in "".join(ch if ch.isalnum() else " " for ch in project_request.lower()).split() if item]
    filtered = [word for word in words if word not in {"build", "create", "add", "fix", "the", "a", "an"}]
    if not filtered:
        return "project"
    return "-".join(filtered[:4])[:48]
