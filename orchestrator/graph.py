"""LangGraph workflow definition."""

from __future__ import annotations

import os
import re
from pathlib import Path

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
from agents.github_ops_agent import GitHubOpsAgent
from agents.manager_agent import ManagerAgent
from agents.manager_agent import build_missing_fields
from agents.manager_agent import extract_onboarding_fields
from agents.project_analyst_agent import ProjectAnalystAgent
from agents.router_agent import IntentRouterAgent
from core.llm import is_error_response
from core.llm import invoke_prompt
from core.prompts import generic_response_prompt
from core.repository import infer_active_project_name
from schemas.llm_outputs import ValidationResult
from schemas.state import WorkflowState
from tools.approvals import TerminalApprovalManager
from tools.audit import get_audit_log_path
from tools.git_tools import git_has_remote
from tools.git_tools import is_git_repository
from tools.git_tools import get_remote_url


def intent_router_node(state: WorkflowState) -> WorkflowState:
    """Classify a request as generic chat or project-related."""
    result = IntentRouterAgent().run(state["project_request"])
    gathered = {**discover_repo_context(), **extract_onboarding_fields(state["project_request"])}
    return {
        "request_type": result["request_type"],
        "project_status": "routing_complete",
        "gathered_requirements": gathered,
        "active_project": gathered.get("repo_name") or infer_active_project_name(state["project_request"]),
    }


def route_after_intent(state: WorkflowState) -> str:
    """Dispatch to generic response or onboarding intake."""
    return "generic_response_node" if state["request_type"] == "GENERIC_CHAT" else "manager_intake_node"


def generic_response_node(state: WorkflowState) -> WorkflowState:
    """Return a normal assistant response for generic chat."""
    try:
        response = invoke_prompt(
            prompt=generic_response_prompt(),
            variables={"project_request": state["project_request"]},
        )
    except Exception:
        response = "I can help with that. Ask your question again with any details you want me to focus on."
    if is_error_response(response):
        response = "I can help with that. Ask your question again with any details you want me to focus on."
    return {
        "project_status": "finished",
        "final_output": response,
    }


def manager_intake_node(state: WorkflowState) -> WorkflowState:
    """Collect missing onboarding information before any engineering work."""
    manager = ManagerAgent()
    gathered = dict(state.get("gathered_requirements", {}))
    missing_fields = list(state.get("missing_fields", [])) or build_missing_fields(gathered, state["project_request"])
    validation_errors = list(state.get("validation_errors", []))
    if missing_fields or validation_errors:
        gathered = manager.collect_onboarding_details(
            project_request=state["project_request"],
            gathered_requirements=gathered,
            missing_fields=missing_fields,
            validation_errors=validation_errors,
        )
    return {
        "project_status": "intake_complete",
        "gathered_requirements": gathered,
        "missing_fields": [],
        "validation_errors": [],
    }


def github_validation_node(state: WorkflowState) -> WorkflowState:
    """Validate GitHub onboarding details before repo initialization."""
    gathered = dict(state.get("gathered_requirements", {}))
    validation = validate_onboarding(gathered, state["project_request"])
    return {
        "project_status": "validated" if validation.status == "passed" else "validation_failed",
        "repo_mode": gathered.get("repo_mode", ""),
        "github_username": gathered.get("github_username", ""),
        "repo_name": gathered.get("repo_name", ""),
        "repo_visibility": gathered.get("repo_visibility", ""),
        "branch_policy": gathered.get("branch_policy", ""),
        "token_ready": gathered.get("token_ready", "").lower() == "true",
        "repo_ready": validation.repo_ready,
        "missing_fields": validation.missing,
        "validation_errors": [*validation.invalid, validation.message] if validation.message else validation.invalid,
    }


def route_after_validation(state: WorkflowState) -> str:
    """Loop on onboarding until validation passes."""
    if not state.get("repo_ready"):
        return "manager_intake_node"
    if state.get("repo_mode") == "existing" and looks_like_project_query(state["project_request"]):
        return "project_analyst_node"
    return "repo_init_node"


def repo_init_node(state: WorkflowState) -> WorkflowState:
    """Prepare or connect the repository after validation passes."""
    gathered = dict(state.get("gathered_requirements", {}))
    repo_agent = GitHubOpsAgent(TerminalApprovalManager())
    repo_plan = {
        "use_current_repo": True,
        "ensure_main_branch": True,
        "sync_with_remote": git_has_remote(),
        "project_branch": "",
        "notes": f"Repo mode={gathered.get('repo_mode', '')}",
    }
    repo_status = repo_agent.prepare_repository(
        active_project=gathered.get("repo_name") or infer_active_project_name(state["project_request"]),
        repo_plan=repo_plan,
    )
    return {
        "project_status": "repo_ready",
        "repo_status": repo_status,
        "execution_context": {
            "github_username": gathered.get("github_username", ""),
            "repo_name": gathered.get("repo_name", ""),
            "repo_mode": gathered.get("repo_mode", ""),
            "repo_visibility": gathered.get("repo_visibility", ""),
            "branch_policy": gathered.get("branch_policy", ""),
            "token_ready": gathered.get("token_ready", ""),
        },
    }


def manager_planning_node(state: WorkflowState) -> WorkflowState:
    """Create execution strategy only after repository readiness is confirmed."""
    plan = ManagerAgent().plan_execution(
        project_request=state["project_request"],
        gathered_requirements={key: str(value) for key, value in state.get("gathered_requirements", {}).items()},
        repo_context={key: value for key, value in state.get("execution_context", {}).items()},
    )
    lead_statuses = build_initial_lead_statuses()
    worker_statuses = build_initial_worker_statuses()
    team_tasks = plan["team_tasks"]

    if plan["execution_mode"] == "PROJECT_QUERY":
        for team_name in get_team_names():
            lead_statuses[team_name]["status"] = "skipped"
            for worker_key in get_team_worker_keys(team_name):
                worker_statuses[worker_key]["status"] = "skipped"
        return {
            "project_status": "planning_complete",
            "execution_mode": plan["execution_mode"],
            "active_project": str(plan["active_project"]),
            "repo_plan": plan["repo_plan"],
            "team_tasks": team_tasks,
            "lead_statuses": lead_statuses,
            "worker_statuses": worker_statuses,
        }

    for team_name, team_config in team_tasks.items():
        if bool(team_config["needed"]):
            lead_statuses[team_name]["status"] = "queued"
            lead_statuses[team_name]["task"] = str(team_config["task"])
        else:
            lead_statuses[team_name]["status"] = "skipped"
            for worker_key in get_team_worker_keys(team_name):
                worker_statuses[worker_key]["status"] = "skipped"

    return {
        "project_status": "planning_complete",
        "execution_mode": plan["execution_mode"],
        "active_project": str(plan["active_project"]),
        "repo_plan": plan["repo_plan"],
        "team_tasks": team_tasks,
        "lead_statuses": lead_statuses,
        "worker_statuses": worker_statuses,
    }


def route_after_planning(state: WorkflowState) -> list[str] | str:
    """Dispatch to analyst mode or engineering teams."""
    if state["execution_mode"] == "PROJECT_QUERY":
        return "project_analyst_node"
    sends = []
    for team_name in get_team_names():
        team_config = state["team_tasks"].get(team_name, {"needed": False})
        if bool(team_config.get("needed")):
            sends.append(f"{team_name}_lead_plan_node")
    return sends or "final_node"


def project_analyst_node(state: WorkflowState) -> WorkflowState:
    """Answer a project-related repository question without creating tasks."""
    answer = ProjectAnalystAgent().run(state["project_request"])
    return {
        "project_status": "finished",
        "analyst_answers": [answer],
        "final_output": str(answer["answer"]),
    }


def make_team_plan_node(team_name: str):
    """Create a lead planning node for one team."""

    def node(state: WorkflowState) -> WorkflowState:
        team_task = str(state["team_tasks"][team_name]["task"])
        lead_spec = get_lead_spec(team_name)
        worker_specs = [worker for worker in get_all_worker_specs() if worker.team == team_name]
        lead_agent = TeamLeadAgent(lead_spec)
        plan = lead_agent.plan_worker_tasks(
            state["project_request"],
            state["execution_mode"],
            state["active_project"],
            team_task,
            worker_specs,
        )

        assignments = plan["assignments"]
        idle_workers = plan["idle_workers"]
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
            "lead_statuses": {team_name: {"name": lead_spec.name, "status": "planning_complete", "task": team_task}},
            "worker_assignments": assignments,
            "worker_statuses": worker_updates,
            "review_queues": {team_name: []},
            "reviewed_paths": {team_name: []},
            "idle_workers": {team_name: idle_workers},
            "team_review_rounds": {team_name: 0},
        }

    return node


def make_worker_dispatch(team_name: str):
    def dispatch(state: WorkflowState) -> list[str] | str:
        sends: list[str] = []
        for worker_key in get_team_worker_keys(team_name):
            if state["worker_assignments"].get(worker_key):
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
        result = worker_agent.execute(
            project_request=state["project_request"],
            execution_mode=state["execution_mode"],
            active_project=state["active_project"],
            team_task=str(state["team_tasks"][worker_spec.team]["task"]),
            assignment=assignment,
            file_owner=state["file_owner"],
            created_files=state["created_files"],
            approval_manager=TerminalApprovalManager(),
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
            "tool_call_records": ([{"team": result["team"], "worker": result["worker_name"], "tool_calls": result["tool_calls"]}] if result["tool_calls"] else []),
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
        worker_outputs = {worker_key: state["worker_outputs"][worker_key] for worker_key in assignments if worker_key in state["worker_outputs"]}
        decisions = lead_agent.review_worker_outputs(
            project_request=state["project_request"],
            execution_mode=state["execution_mode"],
            active_project=state["active_project"],
            team_task=str(state["team_tasks"][team_name]["task"]),
            assignments=assignments,
            worker_outputs=worker_outputs,
            review_round=review_round,
        )

        worker_statuses: dict[str, dict[str, str]] = {}
        review_comments: dict[str, dict[str, object]] = {}
        approved_outputs: dict[str, dict[str, object]] = {}
        for worker_key, decision in decisions.items():
            assignment = assignments[worker_key]
            worker_spec = get_worker_spec_by_key(worker_key)
            status = str(decision["status"])
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

        return {
            "lead_statuses": {team_name: {"name": lead_spec.name, "status": "reviewing", "task": str(state["team_tasks"][team_name]["task"]) }},
            "review_comments": review_comments,
            "approved_outputs": approved_outputs,
            "worker_statuses": worker_statuses,
            "team_review_rounds": {team_name: review_round + 1},
        }

    return node


def make_review_dispatch(team_name: str):
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
        approved_outputs = {
            worker_key: output
            for worker_key, output in state["approved_outputs"].items()
            if str(output.get("team")) == team_name
        }
        team_output = lead_agent.merge_worker_outputs(team_task, approved_outputs)
        return {
            "lead_statuses": {team_name: {"name": lead_spec.name, "status": "finished", "task": team_task}},
            "final_team_outputs": {team_name: {"lead": lead_spec.name, "task": team_task, "output": team_output}},
            "lead_outputs": [{"team": team_name, "lead": lead_spec.name, "task": team_task, "output": team_output}],
        }

    return node


def final_node(state: WorkflowState) -> WorkflowState:
    """Merge lead outputs and produce the final organization output."""
    if state.get("final_output"):
        return state

    ordered_team_outputs = [
        (team_name, state["final_team_outputs"][team_name])
        for team_name in get_team_names()
        if team_name in state["final_team_outputs"]
    ]
    merged_sections = [f"{team_name.title()} Team:\nTask: {item['task']}\n{item['output']}" for team_name, item in ordered_team_outputs]
    merged_output = "\n\n".join(merged_sections)
    final_output = IntegrationAgent().run(
        project_request=state["project_request"],
        execution_mode=state.get("execution_mode", ""),
        active_project=state.get("active_project", ""),
        merged_output=merged_output,
    )
    return {"project_status": "finished", "merged_output": merged_output, "final_output": final_output}


def build_graph():
    """Compile the onboarding-first engineering workflow."""
    workflow = StateGraph(WorkflowState)
    workflow.add_node("intent_router_node", intent_router_node)
    workflow.add_node("generic_response_node", generic_response_node)
    workflow.add_node("manager_intake_node", manager_intake_node)
    workflow.add_node("github_validation_node", github_validation_node)
    workflow.add_node("repo_init_node", repo_init_node)
    workflow.add_node("manager_planning_node", manager_planning_node)
    workflow.add_node("project_analyst_node", project_analyst_node)
    workflow.add_node("final_node", final_node)

    for team_name in get_team_names():
        workflow.add_node(f"{team_name}_lead_plan_node", make_team_plan_node(team_name))
        workflow.add_node(f"{team_name}_lead_review_node", make_lead_review_node(team_name))
        workflow.add_node(f"{team_name}_merge_node", make_team_merge_node(team_name))

    for worker_spec in get_all_worker_specs():
        workflow.add_node(worker_spec.node_name, make_worker_node(worker_spec.key))

    workflow.add_edge(START, "intent_router_node")
    workflow.add_conditional_edges("intent_router_node", route_after_intent, ["generic_response_node", "manager_intake_node"])
    workflow.add_edge("generic_response_node", END)
    workflow.add_edge("manager_intake_node", "github_validation_node")
    workflow.add_conditional_edges("github_validation_node", route_after_validation, ["manager_intake_node", "repo_init_node", "project_analyst_node"])
    workflow.add_edge("repo_init_node", "manager_planning_node")
    workflow.add_conditional_edges(
        "manager_planning_node",
        route_after_planning,
        [f"{team}_lead_plan_node" for team in get_team_names()] + ["project_analyst_node", "final_node"],
    )
    workflow.add_edge("project_analyst_node", END)

    for team_name in get_team_names():
        plan_node = f"{team_name}_lead_plan_node"
        review_node = f"{team_name}_lead_review_node"
        merge_node = f"{team_name}_merge_node"
        team_worker_nodes = [f"{worker_key}_node" for worker_key in get_team_worker_keys(team_name)]
        workflow.add_conditional_edges(plan_node, make_worker_dispatch(team_name), team_worker_nodes + [review_node])
        for worker_node in team_worker_nodes:
            workflow.add_edge(worker_node, review_node)
        workflow.add_conditional_edges(review_node, make_review_dispatch(team_name), team_worker_nodes + [merge_node])
        workflow.add_edge(merge_node, "final_node")

    workflow.add_edge("final_node", END)
    return workflow.compile()


def run_graph(project_request: str) -> WorkflowState:
    """Execute the graph and return the final workflow state."""
    app = build_graph()
    return app.invoke(initial_state(project_request))


def initial_state(project_request: str) -> WorkflowState:
    """Return the default graph state."""
    return {
        "project_request": project_request,
        "request_type": "",
        "execution_mode": "",
        "active_project": "",
        "project_status": "pending",
        "repo_mode": "",
        "github_username": "",
        "repo_name": "",
        "repo_visibility": "",
        "branch_policy": "",
        "token_ready": False,
        "repo_ready": False,
        "validation_errors": [],
        "missing_fields": [],
        "gathered_requirements": {},
        "execution_context": {},
        "team_tasks": {},
        "repo_plan": {},
        "repo_status": {},
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
        "analyst_answers": [],
        "audit_log_path": get_audit_log_path(),
        "merged_output": "",
        "final_output": "",
    }


def validate_onboarding(gathered: dict[str, str], project_request: str) -> ValidationResult:
    """Validate repository onboarding requirements."""
    missing = build_missing_fields(gathered, project_request)
    invalid: list[str] = []
    message = ""

    repo_name = gathered.get("repo_name", "")
    if repo_name and not re.fullmatch(r"[A-Za-z0-9_.-]+", repo_name):
        invalid.append("repo_name")
        message = "Repo names may contain only letters, numbers, dots, underscores, and hyphens."

    repo_mode = gathered.get("repo_mode", "")
    if repo_mode and repo_mode not in {"new", "existing"}:
        invalid.append("repo_mode")

    visibility = gathered.get("repo_visibility", "")
    if visibility and visibility not in {"public", "private"}:
        invalid.append("repo_visibility")

    token_ready = gathered.get("token_ready", "").lower() == "true"
    if not token_ready:
        missing.append("token_ready")

    if repo_mode == "existing" and not is_git_repository():
        invalid.append("repo_connection")
        message = "The current workspace is not a git repository, so an existing repo cannot be connected."

    if repo_mode == "existing" and not git_has_remote():
        invalid.append("remote_access")
        message = "The current repository has no configured remote, so GitHub connectivity cannot be verified."

    status = "passed" if not missing and not invalid else "failed"
    return ValidationResult(
        status=status,
        missing=dedupe(missing),
        invalid=dedupe(invalid),
        message=message,
        repo_ready=status == "passed",
    )


def dedupe(values: list[str]) -> list[str]:
    """Return a list without duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def looks_like_project_query(project_request: str) -> bool:
    """Return True when the request appears to be repository analysis only."""
    lowered = project_request.lower()
    return any(token in lowered for token in {"explain", "architecture", "which files", "how does", "current project"})


def discover_repo_context() -> dict[str, str]:
    """Infer onboarding defaults from the current git workspace when possible."""
    context: dict[str, str] = {}
    if not is_git_repository():
        return context

    context["repo_mode"] = "existing"
    context["repo_name"] = Path.cwd().name
    remote_url = get_remote_url()
    if remote_url:
        match = re.search(r"[:/]([^/]+)/([^/]+?)(?:\.git)?$", remote_url)
        if match:
            context["github_username"] = match.group(1)
            context["repo_name"] = match.group(2)
    context["repo_visibility"] = "private"
    if any(token in remote_url.lower() for token in {"github.com"}):
        context["token_ready"] = "true" if bool(os.getenv("GITHUB_TOKEN")) else "false"
    context["branch_policy"] = "main"
    return context
