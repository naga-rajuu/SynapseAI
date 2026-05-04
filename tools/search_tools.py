"""Code and file search tools."""

from __future__ import annotations

from pathlib import Path

from tools.file_tools import PROJECT_ROOT
from tools.file_tools import resolve_safe_path


def find_file(name: str, directory: str = ".") -> list[str]:
    """Find files by partial name match."""
    root = resolve_safe_path(directory)
    needle = name.lower()
    return sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in root.rglob("*")
        if path.is_file() and needle in path.name.lower()
    )


def grep_keyword(keyword: str, directory: str = ".") -> list[str]:
    """Search text files for a keyword."""
    root = resolve_safe_path(directory)
    results: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if keyword.lower() in line.lower():
                results.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}: {line.strip()}")
    return results[:100]


def search_code(query: str, directory: str = ".") -> list[str]:
    """Alias for keyword search across code files."""
    return grep_keyword(query, directory=directory)


def dependency_lookup(name: str) -> list[str]:
    """Look up a dependency name in requirements and imports."""
    results: list[str] = []
    requirements_path = PROJECT_ROOT / "requirements.txt"
    if requirements_path.exists():
        for line_number, line in enumerate(requirements_path.read_text(encoding="utf-8").splitlines(), start=1):
            if name.lower() in line.lower():
                results.append(f"requirements.txt:{line_number}: {line.strip()}")

    for path in PROJECT_ROOT.rglob("*.py"):
        if any(part in {".venv", ".deps", "__pycache__"} for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if name.lower() in line.lower():
                results.append(f"{path.relative_to(PROJECT_ROOT)}:{line_number}: {line.strip()}")

    return results[:100]
