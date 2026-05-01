"""LangGraph workflow definition."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.dev_agent import DevAgent
from agents.manager_agent import ManagerAgent
from core.llm import is_error_response
from schemas.state import WorkflowState


def manager_agent_node(state: WorkflowState) -> WorkflowState:
    """Create a short implementation plan from the user request."""
    agent = ManagerAgent()
    task_breakdown = agent.run(state["user_input"])

    if is_error_response(task_breakdown):
        return {
            "user_input": state["user_input"],
            "task_breakdown": task_breakdown,
            "dev_output": "Skipped because manager planning failed.",
            "final_output": task_breakdown,
        }

    return {
        "user_input": state["user_input"],
        "task_breakdown": task_breakdown,
        "dev_output": "",
        "final_output": "",
    }


def dev_agent_node(state: WorkflowState) -> WorkflowState:
    """Run the developer agent and return the updated graph state."""
    if is_error_response(state["task_breakdown"]):
        return state

    agent = DevAgent()
    dev_output = agent.run(state["task_breakdown"])

    if is_error_response(dev_output):
        final_output = f"Manager plan created, but developer execution failed.\n{dev_output}"
    else:
        final_output = dev_output

    return {
        "user_input": state["user_input"],
        "task_breakdown": state["task_breakdown"],
        "dev_output": dev_output,
        "final_output": final_output,
    }


def build_graph():
    """Compile the manager-to-developer LangGraph workflow."""
    workflow = StateGraph(WorkflowState)
    workflow.add_node("manager_agent", manager_agent_node)
    workflow.add_node("dev_agent", dev_agent_node)
    workflow.add_edge(START, "manager_agent")
    workflow.add_edge("manager_agent", "dev_agent")
    workflow.add_edge("dev_agent", END)
    return workflow.compile()


def run_graph(user_input: str) -> WorkflowState:
    """Execute the graph and return the final workflow state."""
    app = build_graph()
    result = app.invoke(
        {
            "user_input": user_input,
            "task_breakdown": "",
            "dev_output": "",
            "final_output": "",
        }
    )
    return result
