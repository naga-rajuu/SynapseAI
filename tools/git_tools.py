"""Safe git tool wrappers."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_git_command(args: list[str]) -> str:
    """Run a git command in the project root and return stdout or stderr."""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "git command failed")
    return completed.stdout.strip() or "Command completed successfully."


def git_status() -> str:
    return run_git_command(["status", "--short", "--branch"])


def git_branch() -> str:
    return run_git_command(["branch"])


def git_checkout(branch: str) -> str:
    return run_git_command(["checkout", branch])


def git_add(paths: list[str] | None = None) -> str:
    return run_git_command(["add", *(paths or ["."])])


def git_commit(message: str) -> str:
    return run_git_command(["commit", "-m", message])


def git_diff(target: str | None = None) -> str:
    args = ["diff"]
    if target:
        args.append(target)
    return run_git_command(args)


def git_push(remote: str = "origin", branch: str = "main") -> str:
    return run_git_command(["push", remote, branch])


def git_log(limit: int = 10) -> str:
    return run_git_command(["log", f"-{limit}", "--oneline"])
