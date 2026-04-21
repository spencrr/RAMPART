# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path

import pytest

from tools.rampart_lint.__main__ import main


class TestSubcommandCheck:
    def test_no_args_returns_2(self) -> None:
        assert main([]) == 2

    def test_clean_file_returns_0(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.py"
        f.write_text("async def go_async(): ...\n")
        assert main(["check", str(f)]) == 0

    def test_violations_returns_1(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("async def go(): ...\n")
        assert main(["check", str(f)]) == 1

    def test_directory_input(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "a.py").write_text("async def go(): ...\n")
        (sub / "b.py").write_text("async def go_async(): ...\n")
        assert main(["check", str(sub)]) == 1

    def test_non_python_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        assert main(["check", str(f)]) == 0


class TestSubcommandList:
    def test_returns_0(self) -> None:
        assert main(["list"]) == 0

    def test_prints_all_codes(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["list"])
        output = capsys.readouterr().out
        assert "RAMPART001" in output
        assert "RAMPART002" in output


class TestSubcommandExplain:
    def test_known_code_returns_0(self) -> None:
        assert main(["explain", "RAMPART001"]) == 0

    def test_known_code_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["explain", "RAMPART001"])
        output = capsys.readouterr().out
        assert "RAMPART001" in output
        assert "AsyncSuffixRule" in output
        assert "AsyncFunctionDef" in output

    def test_unknown_code_returns_2(self) -> None:
        assert main(["explain", "RAM999"]) == 2
