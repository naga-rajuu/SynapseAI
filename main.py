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
    print(f"Project Status: {result['project_status']}")
    print("Team Tasks:")
    for team_name, task in result["team_tasks"].items():
        if bool(task["needed"]):
            print(f"{team_name.title()} Team: {task['task']}")

    print("\nLead Status:")
    for team_name, status in result["lead_statuses"].items():
        print(f"{status['name']}: {status['status']}")

    print("\nWorker Status:")
    for team_name, workers in result["worker_statuses"].items():
        for worker_name, worker_status in workers.items():
            print(f"{worker_name}: {worker_status['status']}")

    print("\nLead Outputs:")
    for item in sorted(result["lead_outputs"], key=lambda value: value["team"]):
        print(f"{item['lead']}:")
        print(item["output"])
        print()

    print("Merged Output:")
    print(result["merged_output"])
    print("\nFinal Output:")
    print(result["final_output"])


if __name__ == "__main__":
    main()
