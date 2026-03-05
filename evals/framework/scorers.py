"""Scoring functions for eval output and tool call assertions."""

from __future__ import annotations

import re
from typing import Any, Literal

from evals.framework.schema import ExpectedToolCall, ScorerSpec


def score_output(final_output: str, assertions: list[ScorerSpec]) -> list[str]:
    """Check the agent's final output against a list of scorer specs.

    Args:
        final_output: The last assistant message content.
        assertions: List of scorer specs to evaluate.

    Returns:
        List of failure messages; empty if all assertions pass.
    """
    failures: list[str] = []
    for spec in assertions:
        match spec.kind:
            case "not_empty":
                if not final_output.strip():
                    failures.append("Output assertion failed: expected non-empty output")
            case "contains":
                haystack = final_output if spec.case_sensitive else final_output.lower()
                needle = spec.value if spec.case_sensitive else spec.value.lower()
                if needle not in haystack:
                    failures.append(
                        f"Output assertion failed: expected output to contain {spec.value!r}"
                    )
            case "not_contains":
                haystack = final_output if spec.case_sensitive else final_output.lower()
                needle = spec.value if spec.case_sensitive else spec.value.lower()
                if needle in haystack:
                    failures.append(
                        f"Output assertion failed: expected output NOT to contain {spec.value!r}"
                    )
            case "exact":
                if final_output != spec.value:
                    failures.append(
                        f"Output assertion failed: expected {spec.value!r}, got {final_output!r}"
                    )
            case "regex":
                flags = 0 if spec.case_sensitive else re.IGNORECASE
                if not re.search(spec.value, final_output, flags):
                    failures.append(
                        f"Output assertion failed: expected output to match regex {spec.value!r}"
                    )
    return failures


def score_tool_calls(
    actual_calls: list[tuple[str, dict[str, Any]]],
    expected_calls: list[ExpectedToolCall],
    order: Literal["ordered", "unordered"],
) -> list[str]:
    """Check actual tool calls against expected calls.

    For ``"ordered"``: the expected sequence must appear as a subsequence of
    actual calls (intervening calls such as ``think`` are allowed).
    For ``"unordered"``: each expected call must appear at least once.

    Args:
        actual_calls: List of ``(name, arguments)`` pairs in execution order.
        expected_calls: The calls the agent should have made.
        order: Matching strategy — ``"ordered"`` or ``"unordered"``.

    Returns:
        List of failure messages; empty if all checks pass.
    """
    if not expected_calls:
        return []

    actual_names = [name for name, _ in actual_calls]

    if order == "ordered":
        if not _is_subsequence(actual_calls, expected_calls):
            expected_names = [e.name for e in expected_calls]
            return [
                f"Tool call sequence mismatch.\n"
                f"  Expected subsequence: {expected_names}\n"
                f"  Actual sequence:      {actual_names}"
            ]
        return []

    # unordered — each expected call must appear at least once
    failures: list[str] = []
    for exp in expected_calls:
        if not any(_call_matches(exp, name, args) for name, args in actual_calls):
            failures.append(
                f"Expected tool call '{exp.name}' not found in actual calls: {actual_names}"
            )
    return failures


def _call_matches(
    expected: ExpectedToolCall, actual_name: str, actual_args: dict[str, Any]
) -> bool:
    """Return True if an actual call satisfies an expected call spec.

    Args:
        expected: The expected call specification.
        actual_name: Name of the actual tool called.
        actual_args: Arguments passed to the actual tool.

    Returns:
        True if the actual call matches the expectation.
    """
    if expected.name != actual_name:
        return False
    if expected.arguments is None:
        return True
    if expected.argument_match == "exact":
        return actual_args == expected.arguments
    # subset: all expected keys must be present with matching values
    return all(actual_args.get(k) == v for k, v in expected.arguments.items())


def _is_subsequence(
    actual: list[tuple[str, dict[str, Any]]],
    expected: list[ExpectedToolCall],
) -> bool:
    """Return True if *expected* appears as a subsequence of *actual*.

    Args:
        actual: Actual tool calls in order.
        expected: Expected calls that must appear (in order) within actual.

    Returns:
        True if every expected call can be matched in order within actual.
    """
    idx = 0
    for exp in expected:
        matched = False
        while idx < len(actual):
            name, args = actual[idx]
            idx += 1
            if _call_matches(exp, name, args):
                matched = True
                break
        if not matched:
            return False
    return True
