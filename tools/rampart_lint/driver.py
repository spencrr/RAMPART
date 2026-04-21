# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Lint engine: dispatch, suppression, and file checking."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from tools.rampart_lint.rule import LocatedNode
from tools.rampart_lint.rules.rampart001_async_suffix import AsyncSuffixRule
from tools.rampart_lint.rules.rampart002_keyword_only import KeywordOnlyRule

if TYPE_CHECKING:
    from pathlib import Path

    from tools.rampart_lint.rule import Rule

RULES: list[Rule] = [
    AsyncSuffixRule(),
    KeywordOnlyRule(),
]


def build_dispatch(
    rules: list[Rule],
) -> dict[type[LocatedNode], list[Rule]]:
    """Map each AST node type to the rules that inspect it.

    Raises:
        ValueError: If two rules share the same code.
    """
    seen_codes: dict[str, str] = {}
    dispatch: dict[type[LocatedNode], list[Rule]] = {}

    for rule in rules:
        owner = type(rule).__name__
        if rule.code in seen_codes:
            msg = (
                f"Duplicate rule code {rule.code}: {seen_codes[rule.code]} and {owner}"
            )
            raise ValueError(msg)
        seen_codes[rule.code] = owner

        for node_type in rule.node_types:
            dispatch.setdefault(node_type, []).append(rule)

    return dispatch


def is_suppressed(lines: list[str], lineno: int, code: str) -> bool:
    """Check whether a ``# noqa`` comment suppresses the violation.

    Follows ruff semantics: the ``# noqa`` must appear on the same line as
    the violation. Supports both targeted (``# noqa: RAMPART001``) and bare
    (``# noqa``) suppression.

    Args:
        lines: Source lines (0-indexed list).
        lineno: 1-based line number from the AST node.
        code: Rule code to look for (e.g. ``RAMPART001``).
    """
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return False
    line = lines[idx]
    if "# noqa" not in line:
        return False
    # Bare suppression (no colon after the directive) covers all codes.
    noqa_pos = line.index("# noqa")
    after = line[noqa_pos + len("# noqa") :]
    stripped = after.lstrip()
    if not stripped or stripped[0] != ":":
        return True
    # Targeted: check if our code appears after the colon.
    return code in after


def check_file(path: Path) -> list[str]:
    """Parse a file and run all matching rules against its AST."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return [f"{path}:0: could not parse file"]

    source_lines = source.splitlines()
    errors: list[str] = []

    dispatch = build_dispatch(RULES)

    for node in ast.walk(tree):
        if not isinstance(node, LocatedNode):
            continue
        rules = dispatch.get(type(node))
        if rules is None:
            continue
        for rule in rules:
            msg = rule.check(node)
            if msg and not is_suppressed(source_lines, node.lineno, rule.code):
                errors.append(f"{path}:{node.lineno}: {rule.code} {msg}")

    return errors
