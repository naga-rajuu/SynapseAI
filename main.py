"""Command-line entrypoint for the minimal LangGraph app."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from orchestrator.graph import run_graph


def read_user_task() -> str:
    """Read a task from CLI args or interactive input."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()

    return input("Enter a development task: ").strip()


def main() -> None:
    """Load configuration, run the graph, and print the result."""
    load_dotenv()

    task = read_user_task()
    if not task:
        print("Error: please provide a non-empty task.")
        return

    response = run_graph(task)
    print(response)


if __name__ == "__main__":
    main()
