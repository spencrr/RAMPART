# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Rule protocol and shared utilities for rampart_lint."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Sequence

# AST nodes that carry source location (lineno, col_offset, etc.).
# Using this instead of ast.AST gives typed access to .lineno
LocatedNode: TypeAlias = ast.stmt | ast.expr


class Rule(Protocol):
    """A single lint rule that checks an AST node."""

    code: str
    description: str
    node_types: Sequence[type[LocatedNode]]

    def check(self, node: LocatedNode) -> str | None:
        """Return an error message if the node violates the rule, else None."""
        ...


def is_dunder(name: str) -> bool:
    """Return True for Python dunder names like ``__init__``."""
    return name.startswith("__") and name.endswith("__")
