# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import textwrap

from tools.rampart_lint.rule import LocatedNode
from tools.rampart_lint.rules.rampart001_async_suffix import AsyncSuffixRule


def _parse_first(source: str) -> LocatedNode:
    """Parse source and return the first function/async-function node."""
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            return node
    msg = "no function found in source"
    raise ValueError(msg)


rule = AsyncSuffixRule()


class TestCatches:
    def test_plain_async_function(self) -> None:
        node = _parse_first("async def fetch(): ...")
        assert rule.check(node) is not None

    def test_private_async_function(self) -> None:
        node = _parse_first("async def _fetch(): ...")
        assert rule.check(node) is not None

    def test_async_in_middle_of_name(self) -> None:
        node = _parse_first("async def _async_helper(): ...")
        result = rule.check(node)
        assert result is not None
        assert "_async_helper" in result

    def test_nested_async_function(self) -> None:
        source = """\
        def outer():
            async def inner(): ...
        """
        tree = ast.parse(textwrap.dedent(source))
        violations = [
            rule.check(n) for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
        ]
        assert any(v is not None for v in violations)


class TestPasses:
    def test_async_suffix(self) -> None:
        node = _parse_first("async def fetch_async(): ...")
        assert rule.check(node) is None

    def test_dunder_aenter(self) -> None:
        node = _parse_first("async def __aenter__(): ...")
        assert rule.check(node) is None

    def test_dunder_aexit(self) -> None:
        node = _parse_first("async def __aexit__(): ...")
        assert rule.check(node) is None

    def test_sync_function_ignored(self) -> None:
        node = _parse_first("def fetch(): ...")
        assert rule.check(node) is None

    def test_name_exactly_underscore_async(self) -> None:
        node = _parse_first("async def _async(): ...")
        assert rule.check(node) is None


class TestMetadata:
    def test_code(self) -> None:
        assert rule.code == "RAMPART001"

    def test_node_types(self) -> None:
        assert ast.AsyncFunctionDef in rule.node_types
