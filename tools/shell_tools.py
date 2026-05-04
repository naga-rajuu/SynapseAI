"""Safe shell and process tool wrappers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / ".runtime"
PROCESS_REGISTRY_PATH = RUNTIME_DIR / "processes.json"


def run_command(command: list[str]) -> str:
    """Run a validated shell command and return its output."""
    validate_command(command)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "command failed")
    return completed.stdout.strip() or "Command completed successfully."


def run_tests() -> str:
    return run_command(["python", "-m", "pytest"])


def run_pytest(target: str = ".") -> str:
    return run_command(["python", "-m", "pytest", target])


def build_project() -> str:
    return run_command(["python", "-m", "compileall", "."])


def install_packages(packages: list[str]) -> str:
    if not packages:
        raise ValueError("No packages provided.")
    return run_command(["python", "-m", "pip", "install", *packages])


def start_server(name: str, command: list[str]) -> str:
    """Start a background process and register it by name."""
    validate_command(command)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    registry = load_process_registry()
    registry[name] = {"pid": process.pid, "command": command}
    PROCESS_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return f"Started process {name} with pid {process.pid}."


def stop_process(name: str) -> str:
    """Stop a background process registered by name."""
    registry = load_process_registry()
    if name not in registry:
        raise ValueError(f"Unknown process name: {name}")

    pid = int(registry[name]["pid"])
    completed = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"] if os.name == "nt" else ["kill", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "failed to stop process")

    registry.pop(name, None)
    PROCESS_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return f"Stopped process {name}."


def validate_command(command: list[str]) -> None:
    """Allow only known-safe command prefixes."""
    if not command:
        raise ValueError("Command cannot be empty.")

    allowed_prefixes = [
        ["python", "-m", "pytest"],
        ["python", "-m", "compileall"],
        ["python", "-m", "pip", "install"],
        ["python", "main.py"],
        ["python", "-m", "uvicorn"],
        ["uvicorn"],
    ]
    if any(command[: len(prefix)] == prefix for prefix in allowed_prefixes):
        return
    raise ValueError(f"Command is not allowed: {' '.join(command)}")


def load_process_registry() -> dict[str, dict[str, object]]:
    """Load the process registry file."""
    if not PROCESS_REGISTRY_PATH.exists():
        return {}
    return json.loads(PROCESS_REGISTRY_PATH.read_text(encoding="utf-8"))
