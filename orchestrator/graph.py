"""LangGraph workflow definition."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.agent_factory import build_initial_lead_statuses
from agents.agent_factory import build_initial_worker_statuses
from agents.agent_factory import build_worker_model_map
from agents.agent_factory import get_all_worker_specs
from agents.agent_factory import get_lead_spec
from agents.agent_factory import get_team_names
from agents.agent_factory import get_team_worker_keys
from agents.agent_factory import get_worker_spec_by_key
from agents.dev_agent import IntegrationAgent
from agents.dev_agent import MAX_REVIEW_ROUNDS
from agents.dev_agent import TeamLeadAgent
from agents.dev_agent import WorkerAgent
from agents.manager_agent import ManagerAgent
from core.llm import is_error_response
from schemas.state import WorkflowState
from tools.approvals import TerminalApprovalManager
from tools.audit import get_audit_log_path


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
            "worker_model_map": build_worker_model_map(),
            "lead_statuses": lead_statuses,
            "worker_statuses": worker_statuses,
            "worker_assignments": {},
            "review_comments": {},
            "approved_outputs": {},
            "review_queues": {},
            "reviewed_paths": {},
            "idle_workers": {},
            "assigned_files": {},
            "created_files": {},
            "file_owner": {},
            "final_team_outputs": {},
            "team_review_rounds": {},
            "worker_attempts": {},
            "tool_call_records": [],
            "worker_outputs": {},
            "lead_outputs": [],
            "audit_log_path": get_audit_log_path(),
            "merged_output": "",
            "final_output": team_tasks,
        }

    for team_name, team_config in team_tasks.items():
        if bool(team_config["needed"]):
            lead_statuses[team_name]["status"] = "queued"
            lead_statuses[team_name]["task"] = str(team_config["task"])
        else:
            lead_statuses[team_name]["status"] = "skipped"
            for worker_key in get_team_worker_keys(team_name):
                worker_statuses[worker_key] = {
                    **worker_statuses[worker_key],
                    "status": "skipped",
                }

    return {
        "project_request": state["project_request"],
        "project_status": "running",
        "team_tasks": team_tasks,
        "worker_model_map": build_worker_model_map(),
        "lead_statuses": lead_statuses,
        "worker_statuses": worker_statuses,
        "worker_assignments": {},
        "review_comments": {},
        "approved_outputs": {},
        "review_queues": {},
        "reviewed_paths": {},
        "idle_workers": {},
        "assigned_files": {},
        "created_files": {},
        "file_owner": {},
        "final_team_outputs": {},
        "team_review_rounds": {},
        "worker_attempts": {},
        "tool_call_records": [],
        "worker_outputs": {},
        "lead_outputs": [],
        "audit_log_path": get_audit_log_path(),
        "merged_output": "",
        "final_output": "",
    }


def dispatch_leads(state: WorkflowState) -> list[str] | str:
    """Fan out manager tasks only to the needed team plan nodes."""
    if state["final_output"]:
        return "final_node"

    sends: list[str] = []
    for team_name in get_team_names():
        team_config = state["team_tasks"].get(team_name, {"needed": False, "task": ""})
        if bool(team_config.get("needed")):
            sends.append(f"{team_name}_lead_plan_node")
    return sends or "final_node"


def make_team_plan_node(team_name: str):
    """Create a lead planning node for one team."""

    def node(state: WorkflowState) -> WorkflowState:
        team_task = str(state["team_tasks"][team_name]["task"])
        lead_spec = get_lead_spec(team_name)
        worker_specs = [worker for worker in get_all_worker_specs() if worker.team == team_name]
        lead_agent = TeamLeadAgent(lead_spec)
        plan = lead_agent.plan_worker_tasks(state["project_request"], team_task, worker_specs)

        assignments = plan["assignments"]
        idle_workers = plan["idle_workers"]
        assigned_files = plan["assigned_files"]
        worker_updates: dict[str, dict[str, str]] = {}
        for worker in worker_specs:
            if worker.key in assignments:
                worker_updates[worker.key] = {
                    "name": worker.name,
                    "team": worker.team,
                    "status": "queued",
                    "subtask": str(assignments[worker.key]["subtask"]),
                }
            else:
                worker_updates[worker.key] = {
                    "name": worker.name,
                    "team": worker.team,
                    "status": "idle",
                    "subtask": "",
                }

        return {
            "lead_statuses": {
                team_name: {
                    "name": lead_spec.name,
                    "status": "planning_complete",
                    "task": team_task,
                }
            },
            "worker_assignments": assignments,
            "worker_statuses": worker_updates,
            "review_queues": {team_name: []},
            "reviewed_paths": {team_name: []},
            "idle_workers": {team_name: idle_workers},
            "assigned_files": assigned_files,
            "file_owner": assigned_files,
            "team_review_rounds": {team_name: 0},
        }

    return node


def make_worker_dispatch(team_name: str):
    """Create a conditional dispatcher from a team plan node to worker nodes."""

    def dispatch(state: WorkflowState) -> list[str] | str:
        sends: list[str] = []
        for worker_key in get_team_worker_keys(team_name):
            assignment = state["worker_assignments"].get(worker_key)
            if assignment:
                sends.append(f"{worker_key}_node")
        return sends or f"{team_name}_lead_review_node"

    return dispatch


def make_worker_node(worker_key: str):
    """Create a worker node for one developer."""

    def node(state: WorkflowState) -> WorkflowState:
        assignment = state["worker_assignments"].get(worker_key)
        if not assignment:
            return {}

        worker_spec = get_worker_spec_by_key(worker_key)
        worker_agent = WorkerAgent(worker_spec)
        attempts = int(state["worker_attempts"].get(worker_key, 0)) + 1
        review_info = state["review_comments"].get(worker_key, {})
        review_comments = review_info.get("comments", []) if isinstance(review_info, dict) else []
        model = state["worker_model_map"].get(worker_key) or None
        result = worker_agent.execute(
            project_request=state["project_request"],
            team_task=str(state["team_tasks"][worker_spec.team]["task"]),
            assignment=assignment,
            file_owner=state["file_owner"],
            created_files=state["created_files"],
            approval_manager=TerminalApprovalManager(),
            model=model,
            review_comments=list(review_comments) if isinstance(review_comments, list) else None,
        )
        return {
            "worker_statuses": {
                worker_key: {
                    "name": worker_spec.name,
                    "team": worker_spec.team,
                    "status": "finished",
                    "subtask": str(assignment["subtask"]),
                }
            },
            "worker_outputs": {worker_key: result},
            "created_files": result["created_files"],
            "worker_attempts": {worker_key: attempts},
            "tool_call_records": (
                [
                    {
                        "team": result["team"],
                        "worker": result["worker_name"],
                        "tool_calls": result["tool_calls"],
                    }
                ]
                if result["tool_calls"]
                else []
            ),
        }

    return node


def make_lead_review_node(team_name: str):
    """Create a lead review node for one team."""

    def node(state: WorkflowState) -> WorkflowState:
        lead_spec = get_lead_spec(team_name)
        lead_agent = TeamLeadAgent(lead_spec)
        assignments = {
            worker_key: assignment
            for worker_key, assignment in state["worker_assignments"].items()
            if str(assignment.get("team")) == team_name
        }
        review_round = int(state["team_review_rounds"].get(team_name, 0))
        worker_outputs = {
            worker_key: state["worker_outputs"][worker_key]
            for worker_key in assignments
            if worker_key in state["worker_outputs"]
        }
        decisions = lead_agent.review_worker_outputs(
            project_request=state["project_request"],
            team_task=str(state["team_tasks"][team_name]["task"]),
            assignments=assignments,
            worker_outputs=worker_outputs,
            review_round=review_round,
            reviewed_paths=state["reviewed_paths"].get(team_name, []),
        )

        worker_statuses: dict[str, dict[str, str]] = {}
        review_comments: dict[str, dict[str, object]] = {}
        approved_outputs: dict[str, dict[str, object]] = {}
        reviewed_paths = list(state["reviewed_paths"].get(team_name, []))
        review_queue = lead_agent.build_review_queue(assignments, worker_outputs)
        for worker_key, decision in decisions.items():
            assignment = assignments[worker_key]
            status = str(decision["status"])
            worker_spec = get_worker_spec_by_key(worker_key)
            worker_statuses[worker_key] = {
                "name": worker_spec.name,
                "team": worker_spec.team,
                "status": status,
                "subtask": str(assignment["subtask"]),
            }
            review_comments[worker_key] = {
                "team": team_name,
                "status": status,
                "comments": decision.get("comments", []),
                "note": str(decision.get("note", "")),
            }
            if status == "approved" and worker_key in worker_outputs:
                approved_outputs[worker_key] = worker_outputs[worker_key]
                for path in worker_outputs[worker_key].get("created_paths", []):
                    normalized = str(path)
                    if normalized not in reviewed_paths:
                        reviewed_paths.append(normalized)

        return {
            "lead_statuses": {
                team_name: {
                    "name": lead_spec.name,
                    "status": "reviewing",
                    "task": str(state["team_tasks"][team_name]["task"]),
                }
            },
            "review_comments": review_comments,
            "approved_outputs": approved_outputs,
            "review_queues": {team_name: review_queue},
            "reviewed_paths": {team_name: reviewed_paths},
            "worker_statuses": worker_statuses,
            "team_review_rounds": {team_name: review_round + 1},
        }

    return node


def make_review_dispatch(team_name: str):
    """Route only affected workers back into the fix loop."""

    def dispatch(state: WorkflowState) -> list[str] | str:
        review_round = int(state["team_review_rounds"].get(team_name, 0))
        sends: list[str] = []
        for worker_key in get_team_worker_keys(team_name):
            comment = state["review_comments"].get(worker_key, {})
            if (
                isinstance(comment, dict)
                and str(comment.get("team")) == team_name
                and str(comment.get("status")) == "needs_fix"
                and review_round <= MAX_REVIEW_ROUNDS
            ):
                sends.append(f"{worker_key}_node")
        return sends or f"{team_name}_merge_node"

    return dispatch


def make_team_merge_node(team_name: str):
    """Create a team merge node that only uses approved outputs."""

    def node(state: WorkflowState) -> WorkflowState:
        lead_spec = get_lead_spec(team_name)
        lead_agent = TeamLeadAgent(lead_spec)
        team_task = str(state["team_tasks"][team_name]["task"])
        assignments = {
            worker_key: assignment
            for worker_key, assignment in state["worker_assignments"].items()
            if str(assignment.get("team")) == team_name
        }
        approved_outputs = {
            worker_key: output
            for worker_key, output in state["approved_outputs"].items()
            if str(output.get("team")) == team_name
        }
        for worker_key in assignments:
            if worker_key not in approved_outputs and worker_key in state["worker_outputs"]:
                approved_outputs[worker_key] = state["worker_outputs"][worker_key]
        team_output = lead_agent.merge_worker_outputs(team_task, approved_outputs)
        return {
            "lead_statuses": {
                team_name: {
                    "name": lead_spec.name,
                    "status": "finished",
                    "task": team_task,
                }
            },
            "final_team_outputs": {
                team_name: {
                    "lead": lead_spec.name,
                    "task": team_task,
                    "output": team_output,
                }
            },
            "lead_outputs": [
                {
                    "team": team_name,
                    "lead": lead_spec.name,
                    "task": team_task,
                    "output": team_output,
                }
            ],
        }

    return node


def final_node(state: WorkflowState) -> WorkflowState:
    """Merge lead outputs and produce the final organization output."""
    if state["final_output"]:
        return state

    ordered_team_outputs = [
        (team_name, state["final_team_outputs"][team_name])
        for team_name in get_team_names()
        if team_name in state["final_team_outputs"]
    ]
    merged_sections = [
        f"{team_name.title()} Team:\nTask: {item['task']}\n{item['output']}"
        for team_name, item in ordered_team_outputs
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
    workflow.add_node("final_node", final_node)

    for team_name in get_team_names():
        plan_node = f"{team_name}_lead_plan_node"
        review_node = f"{team_name}_lead_review_node"
        merge_node = f"{team_name}_merge_node"
        workflow.add_node(plan_node, make_team_plan_node(team_name))
        workflow.add_node(review_node, make_lead_review_node(team_name))
        workflow.add_node(merge_node, make_team_merge_node(team_name))

    for worker_spec in get_all_worker_specs():
        workflow.add_node(worker_spec.node_name, make_worker_node(worker_spec.key))

    workflow.add_edge(START, "manager_agent")
    workflow.add_conditional_edges(
        "manager_agent",
        dispatch_leads,
        [f"{team}_lead_plan_node" for team in get_team_names()] + ["final_node"],
    )

    for team_name in get_team_names():
        plan_node = f"{team_name}_lead_plan_node"
        review_node = f"{team_name}_lead_review_node"
        merge_node = f"{team_name}_merge_node"
        team_worker_nodes = [f"{worker_key}_node" for worker_key in get_team_worker_keys(team_name)]
        workflow.add_conditional_edges(
            plan_node,
            make_worker_dispatch(team_name),
            team_worker_nodes + [review_node],
        )
        for worker_node in team_worker_nodes:
            workflow.add_edge(worker_node, review_node)
        workflow.add_conditional_edges(
            review_node,
            make_review_dispatch(team_name),
            team_worker_nodes + [merge_node],
        )
        workflow.add_edge(merge_node, "final_node")

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
            "worker_model_map": build_worker_model_map(),
            "lead_statuses": build_initial_lead_statuses(),
            "worker_statuses": build_initial_worker_statuses(),
            "worker_assignments": {},
            "review_comments": {},
            "approved_outputs": {},
            "review_queues": {},
            "reviewed_paths": {},
            "idle_workers": {},
            "assigned_files": {},
            "created_files": {},
            "file_owner": {},
            "final_team_outputs": {},
            "team_review_rounds": {},
            "worker_attempts": {},
            "tool_call_records": [],
            "worker_outputs": {},
            "lead_outputs": [],
            "audit_log_path": get_audit_log_path(),
            "merged_output": "",
            "final_output": "",
        }
    )
    return result
