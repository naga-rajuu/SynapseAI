"""Command-line entrypoint for the minimal LangGraph app."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from agents.agent_factory import get_team_names
from agents.agent_factory import get_team_worker_keys
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
    for team_name in get_team_names():
        for worker_key in get_team_worker_keys(team_name):
            worker_status = result["worker_statuses"][worker_key]
            line = f"{worker_status['name']}: {worker_status['status']}"
            if worker_status["subtask"]:
                line += f" | {worker_status['subtask']}"
            print(line)

    print("\nIdle Workers:")
    for team_name, workers in result["idle_workers"].items():
        if workers:
            print(f"{team_name.title()}: {', '.join(workers)}")

    print("\nReview Comments:")
    pending_comments = False
    for worker_key, review in result["review_comments"].items():
        if review["comments"]:
            pending_comments = True
            worker_name = result["worker_statuses"][worker_key]["name"]
            print(f"{worker_name}: {'; '.join(review['comments'])}")
    if not pending_comments:
        print("None")

    print("\nTool Calls:")
    if result["tool_call_records"]:
        for record in result["tool_call_records"]:
            print(f"{record['worker']}:")
            for tool_call in record["tool_calls"]:
                status = "success" if bool(tool_call["success"]) else "failed"
                print(f"  - {tool_call['tool']}: {status}")
    else:
        print("None")

    print("\nLead Outputs:")
    for item in sorted(result["lead_outputs"], key=lambda value: value["team"]):
        print(f"{item['lead']}:")
        print(item["output"])
        print()

    print("Merged Output:")
    print(result["merged_output"])
    print("\nFinal Output:")
    print(result["final_output"])
    print(f"\nAudit Log: {result['audit_log_path']}")


if __name__ == "__main__":
    main()
