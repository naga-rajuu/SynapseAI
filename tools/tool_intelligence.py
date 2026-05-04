"""Helpers for inferring tool requests from worker-local subtasks."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
from typing import Any

from core.llm import generate_response
from core.llm import is_error_response


def infer_tool_requests(
    project_request: str,
    team_task: str,
    subtask: str,
    allowed_tools: list[str],
    assigned_files: list[str] | None = None,
    all_team_files: list[str] | None = None,
    setup_folders: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Infer a small set of tool requests for a local subtask."""
    assigned_requests = infer_assignment_requests(
        project_request=project_request,
        team_task=team_task,
        subtask=subtask,
        assigned_files=assigned_files or [],
        all_team_files=all_team_files or assigned_files or [],
        setup_folders=setup_folders or [],
    )
    if assigned_requests:
        return [item for item in assigned_requests if item["tool"] in allowed_tools][:4]

    heuristic_requests = infer_with_heuristics(subtask)
    if heuristic_requests:
        return [item for item in heuristic_requests if item["tool"] in allowed_tools][:4]

    raw_response = generate_response(
        build_tool_prompt(project_request, team_task, subtask, allowed_tools)
    )
    if is_error_response(raw_response):
        return []

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    tool_requests: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", ""))
        params = item.get("params", {})
        if tool in allowed_tools and isinstance(params, dict):
            tool_requests.append({"tool": tool, "params": params})
    return tool_requests[:4]


def build_tool_prompt(
    project_request: str,
    team_task: str,
    subtask: str,
    allowed_tools: list[str],
) -> str:
    """Create a minimal tool-planning prompt."""
    tool_list = ", ".join(allowed_tools)
    return (
        "Choose deterministic tool calls for the local subtask.\n"
        "Return only a JSON array.\n"
        "Each item must have keys: tool and params.\n"
        "Use at most 2 tool calls.\n"
        "If no tool is needed, return [].\n\n"
        f"Allowed tools: {tool_list}\n"
        f"Project request: {project_request}\n"
        f"Team task: {team_task}\n"
        f"Subtask: {subtask}"
    )


def infer_assignment_requests(
    project_request: str,
    team_task: str,
    subtask: str,
    assigned_files: list[str],
    all_team_files: list[str],
    setup_folders: list[str],
) -> list[dict[str, Any]]:
    """Generate tool calls directly from planned file ownership."""
    if not assigned_files:
        return []

    requests: list[dict[str, Any]] = []
    for directory in setup_folders:
        requests.append({"tool": "create_folder", "params": {"path": directory}})

    for path in assigned_files:
        requests.append(
            {
                "tool": "write_file",
                "params": {
                    "path": path,
                    "content": build_file_content(
                        project_request=project_request,
                        team_task=team_task,
                        subtask=subtask,
                        path=path,
                        all_team_files=all_team_files,
                    ),
                },
            }
        )

    return requests[:4]


def infer_with_heuristics(subtask: str) -> list[dict[str, Any]]:
    """Use simple heuristics before asking the model."""
    normalized = subtask.lower()
    file_match = re.search(r"([\w./-]+\.[A-Za-z0-9]+)", subtask)

    if "test" in normalized or "pytest" in normalized:
        return [{"tool": "run_pytest", "params": {"target": "."}}]

    if any(word in normalized for word in ("search", "find", "locate", "review")):
        keyword = extract_keyword(subtask)
        return [{"tool": "search_code", "params": {"query": keyword, "directory": "."}}]

    if any(word in normalized for word in ("create", "write", "boilerplate")) and file_match:
        return [
            {
                "tool": "write_file",
                "params": {
                    "path": file_match.group(1),
                    "content": infer_file_content(subtask, file_match.group(1)),
                },
            }
        ]

    if "dependency" in normalized or "package" in normalized:
        return [{"tool": "dependency_lookup", "params": {"name": extract_keyword(subtask)}}]

    if "delete" in normalized and file_match:
        return [{"tool": "delete_file", "params": {"path": file_match.group(1)}}]

    if "append" in normalized and file_match:
        return [{"tool": "append_file", "params": {"path": file_match.group(1), "content": "\n"}}]

    return []


def extract_keyword(subtask: str) -> str:
    """Extract a simple keyword for search-oriented tools."""
    words = re.findall(r"[A-Za-z0-9_-]+", subtask)
    if not words:
        return "project"
    meaningful = [
        word
        for word in words
        if word.lower() not in {"find", "search", "review", "the", "for", "and"}
    ]
    return meaningful[-1] if meaningful else words[-1]


def infer_file_content(subtask: str, path: str) -> str:
    """Infer a tiny starter file when a file write is requested."""
    normalized = subtask.lower()
    if path.endswith(".py") and "hello world" in normalized:
        return 'print("hello world")\n'
    if path.endswith(".py"):
        return 'print("generated file")\n'
    if path.endswith(".md"):
        return "# Generated File\n"
    return ""


def infer_app_name(project_request: str) -> str:
    """Create a simple folder-safe application name."""
    words = re.findall(r"[A-Za-z0-9]+", project_request.lower())
    filtered = [
        word
        for word in words
        if word not in {"build", "create", "only", "the", "an", "a", "can", "you"}
    ]
    base = "_".join(filtered[:3]) if filtered else "generated_app"
    return base[:40]


def build_file_content(
    project_request: str,
    team_task: str,
    subtask: str,
    path: str,
    all_team_files: list[str],
) -> str:
    """Build deterministic content for an assigned file."""
    normalized = path.replace("\\", "/")
    siblings = {PurePosixPath(item).name for item in all_team_files}
    request_lower = project_request.lower()

    if normalized.endswith("index.html"):
        include_css = "styles.css" in siblings
        include_js = "app.js" in siblings
        return build_frontend_html(project_request, include_css=include_css, include_js=include_js)
    if normalized.endswith("styles.css"):
        return build_frontend_styles(project_request)
    if normalized.endswith("app.js"):
        return build_frontend_script(project_request)
    if normalized.endswith("README.md"):
        return build_readme(project_request, team_task, subtask)
    if normalized.endswith("app.py"):
        return build_backend_entrypoint(project_request)
    if normalized.endswith("services.py"):
        return build_backend_services(project_request)
    if normalized.endswith("schemas.py"):
        return build_backend_schemas(project_request)
    if normalized.endswith(".md"):
        return build_readme(project_request, team_task, subtask)
    if normalized.endswith(".yml") or normalized.endswith(".yaml"):
        return "name: generated\n"
    if normalized.endswith("Dockerfile"):
        return "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nCMD [\"python\", \"main.py\"]\n"
    return infer_file_content(subtask, path)


def build_frontend_html(
    project_request: str,
    include_css: bool,
    include_js: bool,
) -> str:
    """Build a minimal standalone HTML UI."""
    request_lower = project_request.lower()
    title = "Generated UI"
    body = "<main><p>Generated frontend scaffold.</p></main>"

    if "calculator" in request_lower:
        title = "Calculator UI"
        body = """<main class="shell">
  <section class="calculator" aria-label="Calculator">
    <header class="calculator__header">
      <h1>Calculator</h1>
      <p>Clean arithmetic UI</p>
    </header>
    <label class="sr-only" for="display">Calculator display</label>
    <input id="display" class="calculator__display" type="text" readonly value="0" />
    <div class="calculator__keys" role="group" aria-label="Calculator keys">
      <button data-value="7">7</button>
      <button data-value="8">8</button>
      <button data-value="9">9</button>
      <button data-value="/">/</button>
      <button data-value="4">4</button>
      <button data-value="5">5</button>
      <button data-value="6">6</button>
      <button data-value="*">*</button>
      <button data-value="1">1</button>
      <button data-value="2">2</button>
      <button data-value="3">3</button>
      <button data-value="-">-</button>
      <button data-action="clear">C</button>
      <button data-value="0">0</button>
      <button data-action="equals">=</button>
      <button data-value="+">+</button>
    </div>
  </section>
</main>"""

    style_link = '  <link rel="stylesheet" href="./styles.css" />\n' if include_css else ""
    script_tag = '  <script src="./app.js"></script>\n' if include_js else ""
    inline_style = "" if include_css else "  <style>body{font-family:Arial,sans-serif;margin:2rem;}</style>\n"
    inline_script = "" if include_js else "  <script>console.log('Generated UI ready');</script>\n"

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="UTF-8" />\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
        f"  <title>{title}</title>\n"
        f"{style_link}"
        f"{inline_style}"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        f"{script_tag}"
        f"{inline_script}"
        "</body>\n"
        "</html>\n"
    )


def build_frontend_styles(project_request: str) -> str:
    """Build CSS for generated frontend scaffolds."""
    if "calculator" in project_request.lower():
        return """* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  place-items: center;
  font-family: "Segoe UI", sans-serif;
  background: linear-gradient(135deg, #eef2ff, #dbeafe);
  color: #0f172a;
}

.shell {
  width: min(100%, 420px);
  padding: 24px;
}

.calculator {
  background: rgba(255, 255, 255, 0.96);
  border-radius: 24px;
  padding: 24px;
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.16);
}

.calculator__header h1,
.calculator__header p {
  margin: 0;
}

.calculator__header p {
  margin-top: 4px;
  color: #475569;
}

.calculator__display {
  width: 100%;
  margin: 18px 0;
  padding: 14px 16px;
  border: 1px solid #cbd5e1;
  border-radius: 16px;
  font-size: 2rem;
  text-align: right;
}

.calculator__keys {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

button {
  min-height: 52px;
  border: none;
  border-radius: 16px;
  background: #1e293b;
  color: white;
  font-size: 1rem;
  cursor: pointer;
}

button:hover {
  background: #0f172a;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
"""
    return "body {\n  font-family: Arial, sans-serif;\n}\n"


def build_frontend_script(project_request: str) -> str:
    """Build JS logic for generated frontend scaffolds."""
    if "calculator" in project_request.lower():
        return """const display = document.getElementById("display");
const keys = document.querySelector(".calculator__keys");

function setDisplay(value) {
  if (display) {
    display.value = value;
  }
}

function appendValue(value) {
  if (!display) {
    return;
  }
  if (display.value === "0" || display.value === "Error") {
    display.value = "";
  }
  display.value += value;
}

function calculate() {
  if (!display) {
    return;
  }
  try {
    const result = Function(`return ${display.value}`)();
    setDisplay(String(result));
  } catch (error) {
    setDisplay("Error");
  }
}

if (keys) {
  keys.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) {
      return;
    }
    const action = target.dataset.action;
    const value = target.dataset.value;
    if (action === "clear") {
      setDisplay("0");
      return;
    }
    if (action === "equals") {
      calculate();
      return;
    }
    if (value) {
      appendValue(value);
    }
  });
}
"""
    return "console.log('Generated frontend logic');\n"


def build_backend_entrypoint(project_request: str) -> str:
    """Build a small backend entrypoint."""
    return (
        f'"""Generated backend stub for: {project_request}."""\n\n'
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n\n"
        '@app.get("/")\n'
        "def root() -> dict[str, str]:\n"
        '    return {"message": "Generated backend stub"}\n'
    )


def build_backend_services(project_request: str) -> str:
    """Build a small service-layer starter."""
    return (
        f'"""Service helpers for: {project_request}."""\n\n'
        "def health_check() -> dict[str, str]:\n"
        '    return {"status": "ok"}\n'
    )


def build_backend_schemas(project_request: str) -> str:
    """Build a small schema starter."""
    return (
        f'"""Schema definitions for: {project_request}."""\n\n'
        "from pydantic import BaseModel\n\n\n"
        "class GeneratedRequest(BaseModel):\n"
        "    name: str\n"
    )


def build_readme(project_request: str, team_task: str, subtask: str) -> str:
    """Build a concise markdown note."""
    return (
        f"# Generated Artifact\n\n"
        f"- Project request: {project_request}\n"
        f"- Team task: {team_task}\n"
        f"- Worker focus: {subtask}\n"
    )
