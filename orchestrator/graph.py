"""LangGraph workflow definition."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents.dev_agent import DevAgent


class GraphState(TypedDict):
    """Shared state passed between graph nodes."""

    task: str
    response: str


def dev_agent_node(state: GraphState) -> GraphState:
    """Run the developer agent and return the updated graph state."""
    agent = DevAgent()
    return {
        "task": state["task"],
        "response": agent.run(state["task"]),
    }


def build_graph():
    """Compile the single-node LangGraph workflow."""
    workflow = StateGraph(GraphState)
    workflow.add_node("dev_agent", dev_agent_node)
    workflow.add_edge(START, "dev_agent")
    workflow.add_edge("dev_agent", END)
    return workflow.compile()


def run_graph(task: str) -> str:
    """Execute the graph for one task and return the final response."""
    app = build_graph()
    result = app.invoke({"task": task, "response": ""})
    return result["response"]
