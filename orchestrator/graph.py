"""LangGraph workflow definition."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.agent_factory import build_initial_lead_statuses
from agents.agent_factory import build_initial_worker_statuses
from agents.agent_factory import get_lead_spec
from agents.agent_factory import get_team_names
from agents.agent_factory import get_worker_specs
from agents.dev_agent import IntegrationAgent
from agents.dev_agent import TeamLeadAgent
from agents.dev_agent import run_parallel_workers
from agents.manager_agent import ManagerAgent
from core.llm import is_error_response
from schemas.state import WorkflowState


def manager_agent_node(state: WorkflowState) -> WorkflowState:
    """Create team-level tasks from the project request."""
    agent = ManagerAgent()
    team_tasks = agent.run(state["project_request"])
    lead_statuses = build_initial_lead_statuses()
    worker_statuses = build_initial_worker_statuses()

    if isinstance(team_tasks, str) and is_error_response(team_tasks):
        return {
            "project_request": state["project_request"],
            "project_status": "failed",
            "team_tasks": {},
            "lead_statuses": lead_statuses,
            "worker_statuses": worker_statuses,
            "worker_outputs": [],
            "lead_outputs": [],
            "merged_output": "",
            "final_output": team_tasks,
        }

    for team_name, team_config in team_tasks.items():
        if bool(team_config["needed"]):
            lead_statuses[team_name]["status"] = "queued"
            lead_statuses[team_name]["task"] = str(team_config["task"])
        else:
            lead_statuses[team_name]["status"] = "skipped"

        for worker_name, worker_status in worker_statuses[team_name].items():
            if bool(team_config["needed"]):
                worker_statuses[team_name][worker_name] = {
                    **worker_status,
                    "status": "queued",
                }
            else:
                worker_statuses[team_name][worker_name] = {
                    **worker_status,
                    "status": "skipped",
                }

    return {
        "project_request": state["project_request"],
        "project_status": "running",
        "team_tasks": team_tasks,
        "lead_statuses": lead_statuses,
        "worker_statuses": worker_statuses,
        "worker_outputs": [],
        "lead_outputs": [],
        "merged_output": "",
        "final_output": "",
    }


def dispatch_leads(state: WorkflowState) -> list[Send] | str:
    """Fan out manager tasks only to the needed team leads."""
    if state["final_output"]:
        return "final_node"

    branch_map = {
        "backend": "backend_lead_node",
        "frontend": "frontend_lead_node",
        "qa": "qa_lead_node",
        "devops": "devops_lead_node",
    }
    sends = []
    for team_name in get_team_names():
        team_config = state["team_tasks"].get(team_name, {"needed": False, "task": ""})
        if bool(team_config.get("needed")):
            sends.append(
                Send(
                    branch_map[team_name],
                    {
                        "project_request": state["project_request"],
                        "project_status": state["project_status"],
                        "team_tasks": state["team_tasks"],
                        "lead_statuses": state["lead_statuses"],
                        "worker_statuses": state["worker_statuses"],
                        "worker_outputs": [],
                        "lead_outputs": [],
                        "merged_output": "",
                        "final_output": "",
                    },
                )
            )

    if sends:
        return sends
    return "final_node"


def run_lead_flow(state: WorkflowState, team_name: str) -> WorkflowState:
    """Run one lead branch, split work, run workers in parallel, and merge outputs."""
    team_task = str(state["team_tasks"][team_name]["task"])
    lead_spec = get_lead_spec(team_name)
    worker_specs = get_worker_specs(team_name)
    lead_agent = TeamLeadAgent(lead_spec)
    assignments = lead_agent.plan_worker_tasks(
        state["project_request"],
        team_task,
        worker_specs,
    )
    worker_outputs = run_parallel_workers(
        state["project_request"],
        team_task,
        worker_specs,
        assignments,
    )
    lead_output = lead_agent.merge_worker_outputs(team_task, worker_outputs)
    worker_statuses = {
        team_name: {
            item["worker"]: {
                "status": "finished",
                "subtask": item["subtask"],
            }
            for item in worker_outputs
        }
    }
    lead_statuses = {
        team_name: {
            "name": lead_spec.name,
            "status": "finished",
            "task": team_task,
        }
    }

    return {
        "lead_statuses": lead_statuses,
        "worker_statuses": worker_statuses,
        "worker_outputs": worker_outputs,
        "lead_outputs": [
            {
                "team": team_name,
                "lead": lead_spec.name,
                "task": team_task,
                "output": lead_output,
            }
        ],
    }


def backend_lead_node(state: WorkflowState) -> WorkflowState:
    """Run the backend lead branch."""
    return run_lead_flow(state, "backend")


def frontend_lead_node(state: WorkflowState) -> WorkflowState:
    """Run the frontend lead branch."""
    return run_lead_flow(state, "frontend")


def qa_lead_node(state: WorkflowState) -> WorkflowState:
    """Run the QA lead branch."""
    return run_lead_flow(state, "qa")


def devops_lead_node(state: WorkflowState) -> WorkflowState:
    """Run the DevOps lead branch."""
    return run_lead_flow(state, "devops")


def final_node(state: WorkflowState) -> WorkflowState:
    """Merge lead outputs and produce the final organization output."""
    if state["final_output"]:
        return state

    ordered_lead_outputs = sorted(
        state["lead_outputs"],
        key=lambda item: get_team_names().index(item["team"]),
    )
    merged_sections = [
        f"{item['team'].title()} Team:\nTask: {item['task']}\n{item['output']}"
        for item in ordered_lead_outputs
    ]
    merged_output = "\n\n".join(merged_sections)
    integration_agent = IntegrationAgent()
    final_output = integration_agent.run(state["project_request"], merged_output)

    return {
        "project_status": "finished",
        "merged_output": merged_output,
        "final_output": final_output,
    }


def build_graph():
    """Compile the parallel engineering organization workflow."""
    workflow = StateGraph(WorkflowState)
    workflow.add_node("manager_agent", manager_agent_node)
    workflow.add_node("backend_lead_node", backend_lead_node)
    workflow.add_node("frontend_lead_node", frontend_lead_node)
    workflow.add_node("qa_lead_node", qa_lead_node)
    workflow.add_node("devops_lead_node", devops_lead_node)
    workflow.add_node("final_node", final_node)
    workflow.add_edge(START, "manager_agent")
    workflow.add_conditional_edges(
        "manager_agent",
        dispatch_leads,
        [
            "backend_lead_node",
            "frontend_lead_node",
            "qa_lead_node",
            "devops_lead_node",
            "final_node",
        ],
    )
    workflow.add_edge("backend_lead_node", "final_node")
    workflow.add_edge("frontend_lead_node", "final_node")
    workflow.add_edge("qa_lead_node", "final_node")
    workflow.add_edge("devops_lead_node", "final_node")
    workflow.add_edge("final_node", END)
    return workflow.compile()


def run_graph(project_request: str) -> WorkflowState:
    """Execute the graph and return the final workflow state."""
    app = build_graph()
    result = app.invoke(
        {
            "project_request": project_request,
            "project_status": "pending",
            "team_tasks": {},
            "lead_statuses": build_initial_lead_statuses(),
            "worker_statuses": build_initial_worker_statuses(),
            "worker_outputs": [],
            "lead_outputs": [],
            "merged_output": "",
            "final_output": "",
        }
    )
    return result
