"""FastAPI server exposing tool execution."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tools.approvals import PreapprovedApprovalManager
from tools.context import ToolExecutionContext
from tools.tool_router import execute_tool

app = FastAPI(title="Local MCP Tool Server")


class ToolRequest(BaseModel):
    """Incoming tool execution request."""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    agent_name: str = "MCP Client"
    role: str = "system"
    team: str = "platform"
    seniority: str = "system"
    approved: bool | None = None


@app.post("/tool/run")
def run_tool(request: ToolRequest) -> dict[str, object]:
    """Run a tool and return the execution result."""
    return execute_tool(
        request.tool,
        request.params,
        context=ToolExecutionContext(
            agent_name=request.agent_name,
            role=request.role,
            team=request.team,
            seniority=request.seniority,
        ),
        approval_manager=PreapprovedApprovalManager(request.approved),
    )
