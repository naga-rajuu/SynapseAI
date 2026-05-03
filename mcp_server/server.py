"""FastAPI server exposing tool execution."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tools.tool_router import execute_tool

app = FastAPI(title="Local MCP Tool Server")


class ToolRequest(BaseModel):
    """Incoming tool execution request."""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


@app.post("/tool/run")
def run_tool(request: ToolRequest) -> dict[str, object]:
    """Run a tool and return the execution result."""
    return execute_tool(request.tool, request.params)
