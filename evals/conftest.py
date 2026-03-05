"""Eval conftest: trigger tool registration before any eval runs."""

import agent.tools  # noqa: F401 — side-effect import populates TOOL_REGISTRY
