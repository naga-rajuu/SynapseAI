"""Reusable agent specs and factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AgentSpec:
    """Configuration for an agent in the engineering organization."""

    name: str
    team: str
    role: str
    seniority: str
    focus: str
    model: str | None = None

    @property
    def key(self) -> str:
        """Return a stable identifier for graph routing and state."""
        return slugify(self.name)

    @property
    def node_name(self) -> str:
        """Return the LangGraph node name for this agent."""
        return f"{self.key}_node"


TEAM_ORDER = ["backend", "frontend", "qa", "devops"]


TEAM_STRUCTURE: dict[str, dict[str, object]] = {
    "backend": {
        "lead": AgentSpec(
            name="Backend Lead",
            team="backend",
            role="lead",
            seniority="lead",
            focus="Backend architecture, API boundaries, and service coordination",
        ),
        "workers": [
            AgentSpec(
                name="Senior Backend Dev 1",
                team="backend",
                role="worker",
                seniority="senior",
                focus="Service architecture and database design",
            ),
            AgentSpec(
                name="Senior Backend Dev 2",
                team="backend",
                role="worker",
                seniority="senior",
                focus="API contracts, optimization, and reviews",
            ),
            AgentSpec(
                name="Junior Backend Dev 1",
                team="backend",
                role="worker",
                seniority="junior",
                focus="Endpoint implementation and boilerplate",
            ),
            AgentSpec(
                name="Junior Backend Dev 2",
                team="backend",
                role="worker",
                seniority="junior",
                focus="Validation, helpers, and integration basics",
            ),
        ],
    },
    "frontend": {
        "lead": AgentSpec(
            name="Frontend Lead",
            team="frontend",
            role="lead",
            seniority="lead",
            focus="Frontend architecture, UX flow, and component strategy",
        ),
        "workers": [
            AgentSpec(
                name="Senior FE Dev 1",
                team="frontend",
                role="worker",
                seniority="senior",
                focus="UI architecture and state management",
            ),
            AgentSpec(
                name="Senior FE Dev 2",
                team="frontend",
                role="worker",
                seniority="senior",
                focus="Performance, accessibility, and reviews",
            ),
            AgentSpec(
                name="Junior FE Dev 1",
                team="frontend",
                role="worker",
                seniority="junior",
                focus="Component implementation and page scaffolding",
            ),
            AgentSpec(
                name="Junior FE Dev 2",
                team="frontend",
                role="worker",
                seniority="junior",
                focus="Styling, forms, and view wiring",
            ),
        ],
    },
    "qa": {
        "lead": AgentSpec(
            name="QA Lead",
            team="qa",
            role="lead",
            seniority="lead",
            focus="Test strategy, quality gates, and regression coverage",
        ),
        "workers": [
            AgentSpec(
                name="Senior QA Engineer",
                team="qa",
                role="worker",
                seniority="senior",
                focus="Test architecture and risk analysis",
            ),
            AgentSpec(
                name="QA Automation Engineer",
                team="qa",
                role="worker",
                seniority="junior",
                focus="API tests, UI smoke tests, and automation basics",
            ),
        ],
    },
    "devops": {
        "lead": AgentSpec(
            name="DevOps Lead",
            team="devops",
            role="lead",
            seniority="lead",
            focus="Deployment architecture, CI/CD, and runtime operations",
        ),
        "workers": [
            AgentSpec(
                name="Senior DevOps Engineer",
                team="devops",
                role="worker",
                seniority="senior",
                focus="Infrastructure, containerization, and release automation",
            ),
            AgentSpec(
                name="Platform Engineer",
                team="devops",
                role="worker",
                seniority="junior",
                focus="Pipeline wiring, environment setup, and observability basics",
            ),
        ],
    },
}


def get_team_names() -> list[str]:
    """Return the configured team execution order."""
    return list(TEAM_ORDER)


def get_lead_spec(team: str) -> AgentSpec:
    """Return the lead spec for a team."""
    return TEAM_STRUCTURE[team]["lead"]  # type: ignore[return-value]


def get_worker_specs(team: str) -> list[AgentSpec]:
    """Return the worker specs for a team."""
    return list(TEAM_STRUCTURE[team]["workers"])  # type: ignore[return-value]


def get_all_worker_specs() -> list[AgentSpec]:
    """Return every worker spec across all teams."""
    all_workers: list[AgentSpec] = []
    for team in get_team_names():
        all_workers.extend(get_worker_specs(team))
    return all_workers


def get_worker_spec_by_key(worker_key: str) -> AgentSpec:
    """Return the worker spec for a stable worker key."""
    for worker in get_all_worker_specs():
        if worker.key == worker_key:
            return worker
    raise KeyError(f"Unknown worker key: {worker_key}")


def get_team_worker_keys(team: str) -> list[str]:
    """Return the worker keys for a team in the configured order."""
    return [worker.key for worker in get_worker_specs(team)]


def build_worker_model_map() -> dict[str, str]:
    """Build the default per-worker model map."""
    return {
        worker.key: (worker.model or "")
        for worker in get_all_worker_specs()
    }


def build_initial_lead_statuses() -> dict[str, dict[str, str]]:
    """Build the default status map for every lead."""
    return {
        team: {
            "name": get_lead_spec(team).name,
            "status": "pending",
            "task": "",
        }
        for team in get_team_names()
    }


def build_initial_worker_statuses() -> dict[str, dict[str, str]]:
    """Build the default status map for every worker."""
    return {
        worker.key: {
            "name": worker.name,
            "team": worker.team,
            "status": "pending",
            "subtask": "",
        }
        for worker in get_all_worker_specs()
    }


def slugify(value: str) -> str:
    """Return a simple slug used for keys and graph node names."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "agent"
