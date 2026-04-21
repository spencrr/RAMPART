# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ast
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.rampart_lint.driver import build_dispatch, check_file, is_suppressed

# -- is_suppressed -------------------------------------------------------------


class TestIsSuppressed:
    def test_same_line_targeted(self) -> None:
        lines = ["async def fetch(): ...  # noqa: RAMPART001"]
        assert is_suppressed(lines, lineno=1, code="RAMPART001") is True

    def test_wrong_code(self) -> None:
        lines = ["async def fetch(): ...  # noqa: RAMPART002"]
        assert is_suppressed(lines, lineno=1, code="RAMPART001") is False

    def test_different_line(self) -> None:
        lines = [
            "# noqa: RAMPART001",
            "async def fetch(): ...",
        ]
        assert is_suppressed(lines, lineno=2, code="RAMPART001") is False

    def test_bare_noqa_suppresses_all(self) -> None:
        lines = ["async def fetch(): ...  # noqa"]
        assert is_suppressed(lines, lineno=1, code="RAMPART001") is True

    def test_bare_noqa_with_trailing_comment(self) -> None:
        lines = ["async def fetch(): ...  # noqa  # some reason"]
        assert is_suppressed(lines, lineno=1, code="RAMPART002") is True

    def test_multi_code_suppression(self) -> None:
        lines = ["async def f(self, a, b): ...  # noqa: RAMPART001, RAMPART002"]
        assert is_suppressed(lines, lineno=1, code="RAMPART001") is True
        assert is_suppressed(lines, lineno=1, code="RAMPART002") is True

    def test_first_line(self) -> None:
        lines = ["async def fetch(): ...  # noqa: RAMPART001"]
        assert is_suppressed(lines, lineno=1, code="RAMPART001") is True

    def test_last_line(self) -> None:
        lines = ["x = 1", "async def fetch(): ...  # noqa: RAMPART001"]
        assert is_suppressed(lines, lineno=2, code="RAMPART001") is True

    def test_empty_lines(self) -> None:
        assert is_suppressed([], lineno=1, code="RAMPART001") is False

    def test_lineno_out_of_range(self) -> None:
        lines = ["x = 1"]
        assert is_suppressed(lines, lineno=99, code="RAMPART001") is False

    def test_noqa_substring_in_string_literal(self) -> None:
        lines = ['x = "# noqa: RAMPART001"']
        # Known limitation: substring match, not comment parse.
        # Documenting current behavior: it WILL suppress (matches ruff).
        assert is_suppressed(lines, lineno=1, code="RAMPART001") is True


# -- build_dispatch ------------------------------------------------------------


class TestBuildDispatch:
    def test_duplicate_codes_raises(self) -> None:
        rule_a = MagicMock(code="RAMPART001", node_types=(ast.AsyncFunctionDef,))
        rule_b = MagicMock(code="RAMPART001", node_types=(ast.FunctionDef,))
        type(rule_a).__name__ = "RuleA"
        type(rule_b).__name__ = "RuleB"

        with pytest.raises(ValueError, match="Duplicate rule code RAMPART001"):
            build_dispatch([rule_a, rule_b])

    def test_empty_rules(self) -> None:
        assert build_dispatch([]) == {}

    def test_two_rules_same_node_type(self) -> None:
        rule_a = MagicMock(
            code="RAMPART001",
            node_types=(ast.AsyncFunctionDef,),
        )
        rule_b = MagicMock(
            code="RAMPART002",
            node_types=(ast.AsyncFunctionDef, ast.FunctionDef),
        )
        dispatch = build_dispatch([rule_a, rule_b])
        assert len(dispatch[ast.AsyncFunctionDef]) == 2
        assert len(dispatch[ast.FunctionDef]) == 1


# -- check_file ----------------------------------------------------------------


class TestCheckFile:
    def test_finds_violations(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("async def fetch(): ...\n")
        errors = check_file(f)
        assert len(errors) == 1
        assert "RAMPART001" in errors[0]

    def test_noqa_suppresses(self, tmp_path: Path) -> None:
        f = tmp_path / "suppressed.py"
        f.write_text("async def fetch(): ...  # noqa: RAMPART001\n")
        assert check_file(f) == []

    def test_syntax_error(self, tmp_path: Path) -> None:
        f = tmp_path / "broken.py"
        f.write_text("def (\n")
        errors = check_file(f)
        assert len(errors) == 1
        assert "could not parse" in errors[0]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        assert check_file(f) == []

    def test_clean_file(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.py"
        f.write_text(
            textwrap.dedent("""\
                async def fetch_async(): ...
                def process(self, *, a, b): ...
            """),
        )
        assert check_file(f) == []

    def test_multiple_violations(self, tmp_path: Path) -> None:
        f = tmp_path / "multi.py"
        f.write_text(
            textwrap.dedent("""\
                async def one(): ...
                async def two(): ...
            """),
        )
        errors = check_file(f)
        assert len(errors) == 2
