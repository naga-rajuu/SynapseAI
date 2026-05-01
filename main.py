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

    user_input = read_user_task()
    if not user_input:
        print("Error: please provide a non-empty task.")
        return

    result = run_graph(user_input)
    print("Manager Plan:")
    print(result["task_breakdown"])
    print("\nDeveloper Output:")
    print(result["dev_output"])
    print("\nFinal Output:")
    print(result["final_output"])


if __name__ == "__main__":
    main()
