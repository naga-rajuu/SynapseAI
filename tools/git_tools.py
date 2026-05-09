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


def run_git_command_allow_failure(args: list[str]) -> tuple[bool, str]:
    """Run a git command without raising on failure."""
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout.strip() or completed.stderr.strip() or "Command completed successfully."
    return completed.returncode == 0, output


def is_git_repository() -> bool:
    success, output = run_git_command_allow_failure(["rev-parse", "--is-inside-work-tree"])
    return success and output == "true"


def git_has_remote(remote: str = "origin") -> bool:
    success, _ = run_git_command_allow_failure(["remote", "get-url", remote])
    return success


def get_remote_url(remote: str = "origin") -> str:
    success, output = run_git_command_allow_failure(["remote", "get-url", remote])
    return output if success else ""


def get_current_branch() -> str:
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])


def git_branch_exists(branch: str) -> bool:
    success, output = run_git_command_allow_failure(["branch", "--list", branch])
    return success and bool(output.strip())


def git_status() -> str:
    return run_git_command(["status", "--short", "--branch"])


def git_branch() -> str:
    return run_git_command(["branch"])


def git_checkout(branch: str) -> str:
    return run_git_command(["checkout", branch])


def git_create_branch(branch: str, from_branch: str = "main") -> str:
    if git_branch_exists(branch):
        return run_git_command(["checkout", branch])
    return run_git_command(["checkout", "-b", branch, from_branch])


def git_add(paths: list[str] | None = None) -> str:
    return run_git_command(["add", *(paths or ["."])])


def git_commit(message: str) -> str:
    return run_git_command(["commit", "-m", message])


def git_diff(target: str | None = None) -> str:
    args = ["diff"]
    if target:
        args.append(target)
    return run_git_command(args)


def git_pull(remote: str = "origin", branch: str = "main") -> str:
    return run_git_command(["pull", remote, branch])


def git_push(remote: str = "origin", branch: str = "main") -> str:
    return run_git_command(["push", remote, branch])


def git_merge(branch: str) -> str:
    return run_git_command(["merge", "--no-edit", branch])


def git_log(limit: int = 10) -> str:
    return run_git_command(["log", f"-{limit}", "--oneline"])
