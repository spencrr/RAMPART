# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""RAMPART002: multi-param functions must use keyword-only ``*`` separator."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from tools.rampart_lint.rule import is_dunder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.rampart_lint.rule import LocatedNode


class KeywordOnlyRule:
    """Functions with >1 non-self/cls param must enforce keyword-only args."""

    code = "RAMPART002"
    description = "function has multiple positional params — use `*`"
    node_types: Sequence[type[LocatedNode]] = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
    )

    def check(self, node: LocatedNode) -> str | None:
        """Return an error if the function allows >1 positional param."""
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        if is_dunder(node.name):
            return None

        positional = list(node.args.posonlyargs) + list(node.args.args)
        non_self = [a for a in positional if a.arg not in ("self", "cls")]

        if len(non_self) > 1:
            return (
                f"function `{node.name}` has {len(non_self)} positional "
                f"params — use `*` to enforce keyword-only arguments"
            )
        return None
