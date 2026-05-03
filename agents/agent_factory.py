"""Reusable agent specs and factory helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    """Configuration for an agent in the engineering organization."""

    name: str
    team: str
    role: str
    seniority: str
    focus: str


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


def build_initial_worker_statuses() -> dict[str, dict[str, dict[str, str]]]:
    """Build the default status map for every worker."""
    return {
        team: {
            worker.name: {
                "status": "pending",
                "subtask": "",
            }
            for worker in get_worker_specs(team)
        }
        for team in get_team_names()
    }
