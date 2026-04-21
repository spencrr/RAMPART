# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import textwrap

from tools.rampart_lint.rule import LocatedNode
from tools.rampart_lint.rules.rampart002_keyword_only import KeywordOnlyRule


def _parse_first(source: str) -> LocatedNode:
    """Parse source and return the first function node."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            return node
    msg = "no function found in source"
    raise ValueError(msg)


rule = KeywordOnlyRule()


class TestCatches:
    def test_two_positional_method(self) -> None:
        node = _parse_first("def f(self, a, b): ...")
        result = rule.check(node)
        assert result is not None
        assert "2" in result

    def test_three_positional_standalone(self) -> None:
        node = _parse_first("def f(a, b, c): ...")
        result = rule.check(node)
        assert result is not None
        assert "3" in result

    def test_cls_plus_two_positional(self) -> None:
        node = _parse_first("def f(cls, a, b): ...")
        assert rule.check(node) is not None

    def test_async_function(self) -> None:
        node = _parse_first("async def f(self, a, b): ...")
        assert rule.check(node) is not None

    def test_positional_only_counted(self) -> None:
        node = _parse_first("def f(a, b, /): ...")
        assert rule.check(node) is not None


class TestPasses:
    def test_single_param_method(self) -> None:
        node = _parse_first("def f(self, a): ...")
        assert rule.check(node) is None

    def test_single_param_standalone(self) -> None:
        node = _parse_first("def f(a): ...")
        assert rule.check(node) is None

    def test_zero_params(self) -> None:
        node = _parse_first("def f(): ...")
        assert rule.check(node) is None

    def test_keyword_only(self) -> None:
        node = _parse_first("def f(self, *, a, b): ...")
        assert rule.check(node) is None

    def test_dunder_init(self) -> None:
        node = _parse_first("def __init__(self, a, b): ...")
        assert rule.check(node) is None

    def test_dunder_eq(self) -> None:
        node = _parse_first("def __eq__(self, other): ...")
        assert rule.check(node) is None

    def test_vararg(self) -> None:
        node = _parse_first("def f(self, *args): ...")
        assert rule.check(node) is None

    def test_kwargs_only(self) -> None:
        node = _parse_first("def f(self, **kwargs): ...")
        assert rule.check(node) is None

    def test_cls_single_keyword_only(self) -> None:
        node = _parse_first("def f(cls, *, a): ...")
        assert rule.check(node) is None


class TestMetadata:
    def test_code(self) -> None:
        assert rule.code == "RAMPART002"

    def test_node_types(self) -> None:
        assert ast.FunctionDef in rule.node_types
        assert ast.AsyncFunctionDef in rule.node_types
