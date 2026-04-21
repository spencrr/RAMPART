# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""RAMPART001: async functions must end with ``_async`` (dunders exempt)."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from tools.rampart_lint.rule import is_dunder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.rampart_lint.rule import LocatedNode


class AsyncSuffixRule:
    """Async functions must end with the ``_async`` suffix."""

    code = "RAMPART001"
    description = "async function missing `_async` suffix"
    node_types: Sequence[type[LocatedNode]] = (ast.AsyncFunctionDef,)

    def check(self, node: LocatedNode) -> str | None:
        """Return an error if the async function lacks the ``_async`` suffix."""
        if not isinstance(node, ast.AsyncFunctionDef):
            return None
        if is_dunder(node.name):
            return None
        if node.name.endswith("_async"):
            return None
        return f"async function `{node.name}` missing `_async` suffix"
