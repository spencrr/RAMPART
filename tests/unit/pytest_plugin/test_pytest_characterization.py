# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Characterization gates for pytest and pytest-xdist integration behavior."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]

import pytest

if TYPE_CHECKING:
    from _pytest.pytester import Pytester, RunResult


pytest_plugins = ["pytester"]

pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).parents[3]
_OBSERVATIONS_FILE = "controller-observations.jsonl"
_REPORT_ATTRIBUTE = "_rampart_report_envelope_json"
_REPORT_SCHEMA = "rampart.report.characterization.v1"
_REPORT_ENVELOPE = json.dumps(
    {
        "payload": {
            "newline": "\n",
            "quote": '"',
            "slash": "\\",
        },
        "schema": _REPORT_SCHEMA,
    },
    sort_keys=True,
    separators=(",", ":"),
)
_CHILD_PLUGIN_ARGS = (
    "-p",
    "rampart.pytest_plugin.plugin",
    "-p",
    "xdist.plugin",
    "-p",
    "no:cacheprovider",
)
_INI = """\
[pytest]
junit_family = legacy
markers =
    slow: marks tests that spawn subprocess pytest runs; deselect with -m 'not slow'
"""
_REPORT_CONFTEST = f"""\
import json
from pathlib import Path

import pytest


_FINALIZED = pytest.StashKey[bool]()
_IS_WORKER = False
_OBSERVATIONS_FILE = {_OBSERVATIONS_FILE!r}
_REPORT_ATTRIBUTE = {_REPORT_ATTRIBUTE!r}
_REPORT_ENVELOPE = {_REPORT_ENVELOPE!r}


def pytest_configure(config):
    global _IS_WORKER
    _IS_WORKER = hasattr(config, "workerinput")


@pytest.fixture
def finalized_fixture(request):
    yield
    request.node.stash[_FINALIZED] = True


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if not _IS_WORKER or report.when != "teardown":
        return report

    assert item.stash.get(_FINALIZED, False) is True
    setattr(report, _REPORT_ATTRIBUTE, _REPORT_ENVELOPE)
    return report


def pytest_runtest_logreport(report):
    if _IS_WORKER or report.when != "teardown":
        return
    if report.nodeid != "test_transport.py::test_report_transport":
        return

    observation = {{
        "envelope": getattr(report, _REPORT_ATTRIBUTE, None),
        "worker_id": getattr(report, "worker_id", ""),
    }}
    with Path(_OBSERVATIONS_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(observation, sort_keys=True) + "\\n")
"""
_REPORT_TEST = """\
def test_report_transport(finalized_fixture, record_property):
    record_property("control", "visible")
"""


@pytest.fixture
def configured_pytester(
    pytester: Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> Pytester:
    """Configure isolated subprocess runs with the source checkout on PYTHONPATH."""
    pythonpath = [str(_REPO_ROOT)]
    if inherited_pythonpath := os.environ.get("PYTHONPATH"):
        pythonpath.append(inherited_pythonpath)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(pythonpath))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeini(_INI)
    return pytester


def _run_pytest(pytester: Pytester, *args: str) -> RunResult:
    return pytester.runpytest_subprocess(*_CHILD_PLUGIN_ARGS, *args)


def _make_report_project(pytester: Pytester) -> None:
    pytester.makeconftest(_REPORT_CONFTEST)
    pytester.makepyfile(test_transport=_REPORT_TEST)


def _load_observations(pytester: Pytester) -> list[dict[str, Any]]:
    path = pytester.path / _OBSERVATIONS_FILE
    assert path.is_file(), "xdist controller wrote no teardown observations"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _assert_worker_observations(observations: list[dict[str, Any]]) -> None:
    assert len(observations) == 2
    worker_ids = [observation["worker_id"] for observation in observations]
    assert all(isinstance(worker_id, str) and worker_id for worker_id in worker_ids)
    assert len(set(worker_ids)) == 2
    assert all(
        observation["envelope"] == _REPORT_ENVELOPE for observation in observations
    )


def _project_marker_names() -> list[str]:
    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    )
    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    return [
        str(marker).split("(", maxsplit=1)[0].split(":", maxsplit=1)[0]
        for marker in markers
    ]


def _marker_count(*, lines: list[str], name: str) -> int:
    prefix = f"@pytest.mark.{name}"
    return sum(line.startswith((f"{prefix}(", f"{prefix}:")) for line in lines)


class TestProjectMarkerConfiguration:
    def test_only_slow_is_statically_registered(self) -> None:
        marker_names = _project_marker_names()

        assert marker_names.count("harm") == 0
        assert marker_names.count("trial") == 0
        assert marker_names.count("slow") == 1

    def test_markers_are_listed_once(
        self,
        configured_pytester: Pytester,
    ) -> None:
        result = _run_pytest(configured_pytester, "--markers")
        assert result.ret == pytest.ExitCode.OK
        marker_lines = [
            line.strip()
            for line in result.outlines
            if line.strip().startswith("@pytest.mark.")
        ]

        for marker_name in ("harm", "trial", "slow"):
            assert _marker_count(lines=marker_lines, name=marker_name) == 1

    def test_strict_markers_accept_project_markers(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_markers="""\
            import pytest


            @pytest.mark.harm("test")
            @pytest.mark.trial(n=1)
            @pytest.mark.slow
            def test_registered_markers():
                pass
            """,
        )

        result = _run_pytest(configured_pytester, "--strict-markers")

        result.assert_outcomes(passed=1)


class TestParserGroup:
    def test_help_lists_one_positioned_rampart_group(
        self,
        configured_pytester: Pytester,
    ) -> None:
        result = _run_pytest(configured_pytester, "--help")
        assert result.ret == pytest.ExitCode.OK
        help_text = "\n".join(result.outlines)
        header = "RAMPART safety testing:"

        assert help_text.count(header) == 1
        assert help_text.index("general:") < help_text.index(header)
        group_block = help_text[help_text.index(header) :].split("\n\n", maxsplit=1)[0]
        assert "--rampart-xdist-max-bytes" in group_block


class TestXdistReportCharacterization:
    def test_teardown_attribute_reaches_controller_from_each_worker(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _make_report_project(configured_pytester)

        result = _run_pytest(
            configured_pytester,
            "-n",
            "2",
            "--dist=each",
        )

        assert result.ret == pytest.ExitCode.OK
        observations = _load_observations(configured_pytester)
        _assert_worker_observations(observations)


class TestJunitReportCharacterization:
    def test_private_attribute_is_not_a_junit_property(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _make_report_project(configured_pytester)
        xml_path = configured_pytester.path / "report.xml"

        result = _run_pytest(
            configured_pytester,
            "-n",
            "2",
            "--dist=each",
            f"--junitxml={xml_path}",
        )

        assert result.ret == pytest.ExitCode.OK
        observations = _load_observations(configured_pytester)
        _assert_worker_observations(observations)
        xml_text = xml_path.read_text(encoding="utf-8")
        properties = {
            (element.get("name"), element.get("value"))
            for element in ET.fromstring(  # ruff: ignore[suspicious-xml-element-tree-usage]
                xml_text,
            ).findall(".//property")
        }
        assert ("control", "visible") in properties
        assert _REPORT_ATTRIBUTE not in {name for name, _ in properties}
        assert _REPORT_ENVELOPE not in {value for _, value in properties}
        assert _REPORT_ATTRIBUTE not in xml_text
        assert _REPORT_SCHEMA not in xml_text
