"""Reusable file tools for the project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_safe_path(path: str) -> Path:
    """Resolve a path and ensure it stays inside the project root."""
    target_path = (PROJECT_ROOT / path).resolve()
    if not str(target_path).startswith(str(PROJECT_ROOT.resolve())):
        raise ValueError("Path is outside the project workspace.")
    return target_path


def write_file(path: str, content: str) -> str:
    """Write text content to a file inside the project workspace."""
    target_path = resolve_safe_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return f"File written: {target_path.relative_to(PROJECT_ROOT)}"


def read_file(path: str) -> str:
    """Read text content from a file inside the project workspace."""
    target_path = resolve_safe_path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return target_path.read_text(encoding="utf-8")


def list_files(directory: str) -> list[str]:
    """List files inside a directory in the project workspace."""
    target_path = resolve_safe_path(directory or ".")
    if not target_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not target_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    return sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in target_path.rglob("*")
        if path.is_file()
    )
