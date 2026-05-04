"""Audit logging for tool execution."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG_PATH = PROJECT_ROOT / "logs" / "tool_audit.jsonl"


def get_audit_log_path() -> str:
    """Return the audit log path as a string."""
    return str(AUDIT_LOG_PATH)


def summarize_output(value: Any) -> str:
    """Create a short string summary for audit storage."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text[:300]


def write_audit_entry(entry: dict[str, Any]) -> None:
    """Append one audit entry to the JSONL log."""
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **entry,
                },
                default=str,
            )
        )
        handle.write("\n")
