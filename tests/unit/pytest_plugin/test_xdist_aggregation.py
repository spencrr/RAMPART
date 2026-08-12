# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Subprocess (``pytester``) tests for cross-worker aggregation under pytest-xdist.

These tests spawn real child pytest sessions via the ``pytester`` fixture to
exercise the full xdist serialization → merge → emission pipeline. They touch
no live external dependency, but each spins up one or more subprocess runs, so
they are marked ``slow`` and can be deselected with ``-m 'not slow'``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET  # ruff: ignore[suspicious-xml-etree-import]

import pytest

from rampart.pytest_plugin._xdist_shadow import REPORT_ATTRIBUTE

if TYPE_CHECKING:
    from _pytest.pytester import Pytester, RunResult


pytest_plugins = ["pytester"]

pytestmark = pytest.mark.slow

_EXPECTED_TRIAL_MARKER_DEPRECATION_MESSAGE = (
    "The clone-based @pytest.mark.trial marker is deprecated and will be removed "
    "in 0.3.0. Migrate to async batching and preserve CI enforcement: "
    "batch = await execute_trials_async(execution_factory=..., adapter=..., "
    "count=..., threshold=...); assert batch."
)


_CONFTEST = """\
from pathlib import Path

import pytest

from rampart.reporting import JsonFileReportSink


_OUT_DIR = Path("rampart_reports").absolute()


@pytest.fixture(scope="session")
def rampart_sinks():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("rampart_report_dir.txt").write_text(str(_OUT_DIR))
    return [JsonFileReportSink(output_dir=_OUT_DIR)]
"""

_EVIDENCE_CONFTEST = """\
import json
from pathlib import Path


_OUT_PATH = Path("evidence_snapshot.json").absolute()


class SnapshotSink:
    async def emit_async(self, *, report):
        result = report.results[0]
        turn = result.turns[0]
        response = turn.response
        snapshot = {
            "summary": result.summary,
            "harm_category": str(result.harm_category),
            "strategy": result.strategy,
            "nodeid": result.metadata["_pytest_nodeid"],
            "test_name": result.metadata["_pytest_test_name"],
            "result_metadata": result.metadata["user"],
            "prompt": turn.request.prompt,
            "payload_content": turn.request.attachments[0].content,
            "payload_id": turn.request.attachments[0].id,
            "payload_metadata": turn.request.attachments[0].metadata,
            "response_text": response.text,
            "response_metadata": response.metadata,
            "tool_name": response.tool_calls[0].name,
            "tool_arguments": response.tool_calls[0].arguments,
            "tool_result": response.tool_calls[0].result,
            "side_effect_kind": response.side_effects[0].kind,
            "side_effect_details": response.side_effects[0].details,
            "eval_evidence": turn.eval_result.evidence,
            "eval_rationale": turn.eval_result.rationale,
            "driver_reasoning": turn.driver_reasoning,
            "injection_payload_id": result.injections[0].payload_id,
            "injection_surface_name": result.injections[0].surface_name,
        }
        _OUT_PATH.write_text(
            json.dumps(snapshot, ensure_ascii=True),
            encoding="utf-8",
        )


def pytest_rampart_sinks(config):
    return [SnapshotSink()]
"""

_SHADOW_SNAPSHOT_CONFTEST = """\
import json
from pathlib import Path

import pytest

from rampart.pytest_plugin._xdist_shadow import (
    REPORT_ATTRIBUTE,
    SHADOW_MANIFEST_KEY,
    SHADOW_RUNTIME_KEY,
)


_MANIFEST_MODE = "__MANIFEST_MODE__"
_REPORT_MODE = "__REPORT_MODE__"
_SHADOW_SNAPSHOT = Path("shadow_snapshot.json").absolute()
_TRIAL_METADATA_KEYS = (
    "_rampart_trial_batch_schema",
    "_rampart_trial_batch_id",
    "_rampart_trial_batch_index",
    "_rampart_trial_batch_count",
    "_rampart_trial_batch_threshold",
)


def _result_record(result):
    metadata = result.metadata
    return {
        "summary": result.summary,
        "worker_id": metadata.get("_rampart_source_worker"),
        "nodeid": metadata.get("_pytest_nodeid"),
        "result_index": metadata.get("_rampart_result_index"),
        "trial_metadata": {
            key: metadata[key] for key in _TRIAL_METADATA_KEYS if key in metadata
        },
    }


def _record_key(record):
    return (
        record["worker_id"] or "",
        record["nodeid"] or "",
        record["result_index"],
        record["summary"],
    )


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    report = yield
    config = item.config
    if (
        hasattr(config, "workerinput")
        and report.when == "teardown"
        and hasattr(report, REPORT_ATTRIBUTE)
        and _REPORT_MODE == "malformed"
    ):
        setattr(report, REPORT_ATTRIBUTE, "{")
    return report


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    del exitstatus
    config = session.config
    if hasattr(config, "workerinput"):
        if _MANIFEST_MODE == "missing":
            config.workeroutput.pop(SHADOW_MANIFEST_KEY, None)
        elif _MANIFEST_MODE == "empty":
            config.workeroutput[SHADOW_MANIFEST_KEY] = {}
        return

    runtime = config.stash.get(SHADOW_RUNTIME_KEY, None)
    if runtime is None:
        return
    deliveries = sorted(
        runtime._controller.deliveries.values(),
        key=lambda item: (item.worker_id, item.envelope.sequence),
    )
    per_worker = {}
    for delivery in deliveries:
        per_worker.setdefault(delivery.worker_id, []).append(
            delivery.envelope.sequence,
        )
    v1_records = sorted(
        (
            _result_record(result)
            for results in runtime.session.results_by_nodeid.values()
            for result in results
        ),
        key=_record_key,
    )
    v2_records = []
    for delivery in deliveries:
        for validated in delivery.envelope.results:
            record = _result_record(validated.result)
            record["slot_index"] = validated.index
            v2_records.append(record)
    v2_records.sort(key=_record_key)
    snapshot = {
        "delivery_count": len(deliveries),
        "faults": sorted(runtime._controller.fault_codes),
        "incomplete": runtime.session.is_incomplete,
        "slot_indexes": [
            [result.index for result in delivery.envelope.results]
            for delivery in deliveries
        ],
        "sequences": sorted(per_worker.values()),
        "shadow_results": sum(
            len(delivery.envelope.results) for delivery in deliveries
        ),
        "summaries": [
            result.result.summary
            for delivery in deliveries
            for result in delivery.envelope.results
        ],
        "v1_results": sum(
            len(results) for results in runtime.session.results_by_nodeid.values()
        ),
        "v1_records": v1_records,
        "v2_records": v2_records,
        "worker_count": len(per_worker),
    }
    _SHADOW_SNAPSHOT.write_text(
        json.dumps(snapshot, sort_keys=True),
        encoding="utf-8",
    )
"""


# Each ``pytester`` child session is configuration-isolated from the repository's
# ``pyproject.toml``, so pytest-asyncio reads an empty
# ``asyncio_default_fixture_loop_scope`` via ``config.getini(...)`` and emits a
# ``PytestDeprecationWarning`` once per subprocess run. Writing an ini file into
# the child project root sets the option through the same channel pytest-asyncio
# reads, mirroring the parent project's asyncio configuration.
_INI = """\
[pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = session
"""


@pytest.fixture
def configured_pytester(pytester: Pytester) -> Pytester:
    """Write the child-session pytest conftest and ini files for subprocess runs."""
    pytester.makeconftest(_CONFTEST)
    pytester.makeini(_INI)
    return pytester


def _load_reports(configured_pytester: Pytester) -> list[dict[str, Any]]:
    marker = configured_pytester.path / "rampart_report_dir.txt"
    if not marker.exists():
        default_dir = configured_pytester.path / "rampart_reports"
        if default_dir.exists():
            return [
                json.loads(p.read_text())
                for p in sorted(default_dir.glob("run_report_*.json"))
            ]
        return []
    out_dir = Path(marker.read_text().strip())
    if not out_dir.exists():
        return []
    return [
        json.loads(p.read_text()) for p in sorted(out_dir.glob("run_report_*.json"))
    ]


def _setup_simple_tests(configured_pytester: Pytester) -> None:
    configured_pytester.makepyfile(
        test_a="""
        import pytest
        from rampart import record_result
        from rampart.core.result import Result, SafetyStatus
        from rampart.core.types import ObservabilityLevel

        @pytest.mark.harm("test")
        def test_a_one():
            record_result(Result(
                status=SafetyStatus.SAFE, summary="a1",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))

        @pytest.mark.harm("test")
        def test_a_two():
            record_result(Result(
                status=SafetyStatus.UNSAFE, summary="a2",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))
        """,
        test_b="""
        import pytest
        from rampart import record_result
        from rampart.core.result import Result, SafetyStatus
        from rampart.core.types import ObservabilityLevel

        @pytest.mark.harm("test")
        def test_b_one():
            record_result(Result(
                status=SafetyStatus.SAFE, summary="b1",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))

        @pytest.mark.harm("test")
        def test_b_two():
            record_result(Result(
                status=SafetyStatus.SAFE, summary="b2",
                observability_level=ObservabilityLevel.RESPONSE_ONLY,
            ))
        """,
    )


def _load_evidence_snapshot(configured_pytester: Pytester) -> dict[str, Any]:
    path = configured_pytester.path / "evidence_snapshot.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_shadow_snapshot(
    configured_pytester: Pytester,
    *,
    manifest_mode: str = "",
    report_mode: str = "",
) -> None:
    observer = _SHADOW_SNAPSHOT_CONFTEST.replace(
        "__MANIFEST_MODE__",
        manifest_mode,
    ).replace(
        "__REPORT_MODE__",
        report_mode,
    )
    configured_pytester.makeconftest(f"{_CONFTEST}\n{observer}")


def _load_shadow_snapshot(configured_pytester: Pytester) -> dict[str, Any]:
    path = configured_pytester.path / "shadow_snapshot.json"
    assert path.is_file(), "controller wrote no v2 shadow snapshot"
    return json.loads(path.read_text(encoding="utf-8"))


def _make_shadow_result_tests(configured_pytester: Pytester) -> None:
    configured_pytester.makepyfile(
        test_shadow="""\
        import pytest

        from rampart import record_result
        from rampart.core.result import Result, SafetyStatus


        @pytest.mark.xdist_group(name="shadow")
        def test_shadow_one():
            record_result(Result(status=SafetyStatus.SAFE, summary="one"))


        @pytest.mark.xdist_group(name="shadow")
        def test_shadow_two():
            record_result(Result(status=SafetyStatus.SAFE, summary="two"))
        """,
    )


def _flatten_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result for results in report["by_harm_category"].values() for result in results
    ]


class TestSingleProcessBaseline:
    def test_baseline_emits_one_report(self, configured_pytester: Pytester) -> None:
        _setup_simple_tests(configured_pytester)
        result = configured_pytester.runpytest("-p", "no:cacheprovider")
        result.assert_outcomes(passed=4)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 4


class TestXdistConsolidation:
    def test_xdist_emits_single_consolidated_report(
        self,
        configured_pytester: Pytester,
    ) -> None:
        """A distributed run merges into one report with full population stats.

        A single ``-n 2`` run proves both that xdist yields exactly one
        consolidated report and that the merged population statistics reflect
        the entire set (per-field aggregation itself is unit-tested in
        ``tests/unit/reporting/test_report.py::TestPopulationSummary``).
        """
        _setup_simple_tests(configured_pytester)
        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
        )
        result.assert_outcomes(passed=4)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1, (
            f"Expected exactly one report under xdist, got {len(reports)}: "
            f"{[r.get('total_runs') for r in reports]}"
        )
        report = reports[0]
        assert report["total_runs"] == 4
        assert report["passed"] == 3
        assert report["failed"] == 1
        assert report["population_summary"]["total_runs"] == 4
        assert report["population_summary"]["safe_count"] == 3
        assert report["population_summary"]["unsafe_count"] == 1


class TestEvidenceParity:
    def test_serial_and_xdist_preserve_identical_textual_evidence(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makeconftest(_EVIDENCE_CONFTEST)
        configured_pytester.makepyfile(
            test_evidence=r"""
            import pytest

            from rampart import record_result
            from rampart.core.result import InjectionRecord, Result, SafetyStatus
            from rampart.core.types import (
                EvalOutcome,
                EvalResult,
                Payload,
                Request,
                Response,
                SideEffect,
                ToolCall,
                Turn,
            )


            def _text(label):
                return f"{label}\x1b[31m\t\n\r\x7f\x9b雪"


            @pytest.mark.harm(_text("category"))
            def test_evidence():
                payload = Payload(
                    content=_text("payload-content"),
                    id=_text("payload-id"),
                    metadata={"nested": _text("payload-metadata")},
                )
                response = Response(
                    text=_text("response"),
                    tool_calls=[
                        ToolCall(
                            name=_text("tool-name"),
                            arguments={"nested": _text("tool-arguments")},
                            result=_text("tool-result"),
                        ),
                    ],
                    side_effects=[
                        SideEffect(
                            kind=_text("side-effect-kind"),
                            details={"nested": _text("side-effect-details")},
                        ),
                    ],
                    metadata={"nested": _text("response-metadata")},
                )
                turn = Turn(
                    request=Request(prompt=_text("prompt"), attachments=[payload]),
                    response=response,
                    eval_result=EvalResult(
                        outcome=EvalOutcome.DETECTED,
                        evidence=[_text("eval-evidence")],
                        rationale=_text("eval-rationale"),
                    ),
                    driver_reasoning=_text("driver-reasoning"),
                )
                result = Result(
                    status=SafetyStatus.SAFE,
                    summary=_text("summary"),
                    turns=[turn],
                    harm_category=_text("result-category"),
                    strategy=_text("strategy"),
                    injections=[
                        InjectionRecord(
                            payload_id=_text("injection-payload-id"),
                            surface_name=_text("injection-surface"),
                        ),
                    ],
                    metadata={"user": {"nested": _text("result-metadata")}},
                )
                record_result(result)
                assert result
            """,
        )

        serial_run = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "--color=no",
            "-q",
        )
        serial_run.assert_outcomes(passed=1)
        serial_snapshot = _load_evidence_snapshot(configured_pytester)
        (configured_pytester.path / "evidence_snapshot.json").unlink()

        xdist_run = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "--color=no",
            "-q",
            "-n",
            "2",
        )
        xdist_run.assert_outcomes(passed=1)
        xdist_snapshot = _load_evidence_snapshot(configured_pytester)

        assert xdist_snapshot == serial_snapshot


class TestXdistTrialAggregation:
    def test_trial_aggregation_across_workers_loadgroup(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_trial="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.5)
            def test_trial_split():
                record_result(Result(
                    status=SafetyStatus.SAFE, summary="t",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "loadgroup",
        )
        result.assert_outcomes(passed=4)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 4

    def test_trial_aggregation_across_workers_load(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_trial="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.5)
            def test_trial_split():
                record_result(Result(
                    status=SafetyStatus.SAFE, summary="t",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "load",
        )
        result.assert_outcomes(passed=4)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 4

    def test_trial_group_fails_when_any_unsafe_under_load(
        self,
        configured_pytester: Pytester,
    ) -> None:
        """Same as above but with --dist=load so clones may split workers.

        The PR docs claim aggregation remains correct under --dist=load
        because the controller merges all worker results. This test
        protects that contract: an UNSAFE clone produced on any worker
        must propagate into the controller's trial-group verdict.
        """
        configured_pytester.makepyfile(
            test_trial_mixed_load="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import ObservabilityLevel

            @pytest.mark.harm("test")
            @pytest.mark.trial(n=4, threshold=0.5)
            def test_trial_mixed_load(request):
                unsafe = request.node.name.endswith("[trial-3]")
                record_result(Result(
                    status=SafetyStatus.UNSAFE if unsafe else SafetyStatus.SAFE,
                    summary="u" if unsafe else "s",
                    observability_level=ObservabilityLevel.RESPONSE_ONLY,
                ))
            """,
        )
        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "load",
        )
        result.assert_outcomes(passed=4)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 4
        assert report["failed"] == 1
        summary = "\n".join(result.outlines)
        assert (
            "FAIL  test_trial_mixed_load [3/4 safe, 75% pass rate, threshold: 50%]"
            in summary
        )


class TestTrialExitStatus:
    def test_default_threshold_fails_below_full_pass_rate(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_below_threshold="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus

            @pytest.mark.trial(n=2)
            def test_below_threshold(request):
                safe = request.node.name.endswith("[trial-0]")
                record_result(Result(
                    status=(
                        SafetyStatus.SAFE if safe else SafetyStatus.UNDETERMINED
                    ),
                    summary="safe" if safe else "undetermined",
                ))
            """,
        )

        result = configured_pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert (
            "FAIL  test_below_threshold "
            "[1/2 safe, 50% pass rate, threshold: 100%]" in "\n".join(result.outlines)
        )

    def test_all_error_group_exits_nonzero(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_all_error="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus

            @pytest.mark.trial(n=2)
            def test_all_error():
                record_result(Result(
                    status=SafetyStatus.ERROR,
                    summary="infrastructure error",
                ))
            """,
        )

        result = configured_pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)
        assert result.ret == pytest.ExitCode.TESTS_FAILED

    def test_all_no_result_group_exits_nonzero(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_all_no_result="""
            import pytest

            @pytest.mark.trial(n=2)
            def test_all_no_result():
                pass
            """,
        )

        result = configured_pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(passed=2)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert (
            "FAIL  test_all_no_result "
            "[0/2 safe, 2 no-result, 0% pass rate, threshold: 100%] -- FAILED"
            in "\n".join(result.outlines)
        )

    def test_invalid_threshold_fails_collection_clearly(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_invalid_threshold="""
            import pytest

            @pytest.mark.trial(n=2, threshold=10**5000)
            def test_invalid_threshold():
                pass
            """,
        )

        result = configured_pytester.runpytest("-p", "no:cacheprovider")

        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*trial(threshold=)*finite number in (0, 1]*"])


class TestXdistEachAttempts:
    def test_each_worker_execution_is_preserved(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_each="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus

            @pytest.mark.harm("test")
            def test_each():
                record_result(Result(status=SafetyStatus.SAFE, summary="kept"))
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "each",
        )

        result.assert_outcomes(passed=2)
        assert result.ret == pytest.ExitCode.OK
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 2
        assert reports[0]["population_summary"]["total_runs"] == 2

    def test_each_trial_retains_attempts_but_gates_logical_clones(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_each_trial="""
            import os

            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus

            @pytest.mark.trial(n=2)
            def test_each_trial():
                worker_id = os.environ["PYTEST_XDIST_WORKER"]
                status = (
                    SafetyStatus.UNSAFE
                    if worker_id.endswith("0")
                    else SafetyStatus.SAFE
                )
                record_result(Result(status=status, summary=status.value))
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "each",
        )

        result.assert_outcomes(passed=4)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 4
        assert reports[0]["passed"] == 2
        assert reports[0]["failed"] == 2
        assert (
            "FAIL  test_each_trial "
            "[0/2 safe, 0% pass rate, threshold: 100%]" in "\n".join(result.outlines)
        )


class TestExecutionDomainTrialBatches:
    def test_one_item_fixture_and_junit_case_yields_n_results(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_execution_trials="""
            from pathlib import Path

            import pytest

            from rampart import (
                BaseExecution,
                Result,
                SafetyStatus,
                execute_trials_async,
            )


            class Execution(BaseExecution):
                @property
                def strategy_name(self):
                    return "test"

                async def _execute_async(self, *, adapter):
                    return Result(status=SafetyStatus.SAFE, summary="safe")


            @pytest.fixture
            def adapter():
                path = Path("fixture-lifecycle.txt")
                path.write_text("setup\\n", encoding="utf-8")
                yield object()
                with path.open("a", encoding="utf-8") as stream:
                    stream.write("teardown\\n")


            @pytest.mark.harm("test")
            async def test_execution_trials(adapter):
                batch = await execute_trials_async(
                    execution_factory=Execution,
                    adapter=adapter,
                    count=3,
                )
                assert batch
            """,
        )
        xml_path = configured_pytester.path / "report.xml"

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml_path}",
        )

        result.assert_outcomes(passed=1)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 3
        assert len(reports[0]["trial_batches"]) == 1
        assert reports[0]["trial_batches"][0]["passed"] is True
        assert (
            len(
                ET.parse(  # ruff: ignore[suspicious-xml-element-tree-usage]
                    xml_path,
                ).findall(".//testcase"),
            )
            == 1
        )
        assert (
            configured_pytester.path / "fixture-lifecycle.txt"
        ).read_text() == "setup\nteardown\n"

    def test_partial_and_zero_factory_failures_preserve_only_real_results(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_factory_failures="""
            from rampart import (
                BaseExecution,
                Result,
                SafetyStatus,
                execute_trials_async,
            )


            class Execution(BaseExecution):
                @property
                def strategy_name(self):
                    return "test"

                async def _execute_async(self, *, adapter):
                    return Result(status=SafetyStatus.SAFE, summary="safe")


            async def test_partial():
                calls = 0

                def factory():
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("partial factory failure")
                    return Execution()

                await execute_trials_async(
                    execution_factory=factory,
                    adapter=object(),
                    count=3,
                )


            async def test_zero():
                def factory():
                    raise RuntimeError("zero factory failure")

                await execute_trials_async(
                    execution_factory=factory,
                    adapter=object(),
                    count=3,
                )
            """,
        )

        result = configured_pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(failed=2)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 1
        assert len(report["trial_batches"]) == 1
        [summary] = report["trial_batches"]
        assert summary["safe_count"] == 1
        assert summary["requested_count"] == 3
        assert summary["complete"] is False
        assert summary["passed"] is False

    def test_complete_subthreshold_batch_fails_item_and_reports_complete(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_failed_gate="""
            from rampart import (
                BaseExecution,
                Result,
                SafetyStatus,
                execute_trials_async,
            )


            class Execution(BaseExecution):
                def __init__(self, *, status):
                    super().__init__()
                    self._status = status

                @property
                def strategy_name(self):
                    return "test"

                async def _execute_async(self, *, adapter):
                    return Result(status=self._status, summary=self._status.value)


            async def test_failed_gate():
                passing = await execute_trials_async(
                    execution_factory=lambda: Execution(status=SafetyStatus.SAFE),
                    adapter=object(),
                    count=1,
                )
                assert passing

                statuses = iter([SafetyStatus.SAFE, SafetyStatus.UNSAFE])
                failing = await execute_trials_async(
                    execution_factory=lambda: Execution(status=next(statuses)),
                    adapter=object(),
                    count=2,
                )
                assert failing
            """,
        )

        result = configured_pytester.runpytest("-p", "no:cacheprovider")

        result.assert_outcomes(failed=1)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 3
        assert len(report["trial_batches"]) == 2
        passing, failing = report["trial_batches"]
        assert passing["passed"] is True
        assert failing["complete"] is True
        assert failing["safe_count"] == 1
        assert failing["unsafe_count"] == 1
        assert failing["passed"] is False

    def test_normal_xdist_keeps_internal_trials_on_one_worker(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_xdist_batch="""
            from rampart import (
                BaseExecution,
                Result,
                SafetyStatus,
                execute_trials_async,
                record_result,
            )


            class Execution(BaseExecution):
                @property
                def strategy_name(self):
                    return "test"

                async def _execute_async(self, *, adapter):
                    return Result(status=SafetyStatus.SAFE, summary="batch")


            async def test_xdist_batch():
                record_result(Result(status=SafetyStatus.SAFE, summary="plain"))
                batch = await execute_trials_async(
                    execution_factory=Execution,
                    adapter=object(),
                    count=2,
                )
                assert batch
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
        )

        result.assert_outcomes(passed=1)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 3
        assert len(report["trial_batches"]) == 1
        [summary] = report["trial_batches"]
        assert summary["safe_count"] == 2
        assert summary["complete"] is True

    def test_dist_each_creates_one_unique_batch_per_worker_invocation(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_each_batch="""
            import pytest

            from rampart import (
                BaseExecution,
                Result,
                SafetyStatus,
                execute_trials_async,
            )


            class Execution(BaseExecution):
                @property
                def strategy_name(self):
                    return "test"

                async def _execute_async(self, *, adapter):
                    return Result(status=SafetyStatus.SAFE, summary="safe")


            @pytest.mark.harm("test")
            async def test_each_batch():
                batch = await execute_trials_async(
                    execution_factory=Execution,
                    adapter=object(),
                    count=2,
                )
                assert batch
            """,
        )
        xml_path = configured_pytester.path / "each-report.xml"

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--dist",
            "each",
            f"--junitxml={xml_path}",
        )

        result.assert_outcomes(passed=2)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == 4
        assert len(report["trial_batches"]) == 2
        batch_ids = [summary["batch_id"] for summary in report["trial_batches"]]
        assert len(set(batch_ids)) == 2
        assert all(
            summary["requested_count"] == 2 for summary in report["trial_batches"]
        )
        assert all(summary["complete"] is True for summary in report["trial_batches"])
        first_seen = list(
            dict.fromkeys(
                result["metadata"]["_rampart_trial_batch_id"]
                for result in _flatten_results(report)
            ),
        )
        assert batch_ids == first_seen
        assert (
            len(
                ET.parse(  # ruff: ignore[suspicious-xml-element-tree-usage]
                    xml_path,
                ).findall(".//testcase"),
            )
            == 2
        )

    @pytest.mark.parametrize("dist_mode", ["load", "loadgroup", "each"])
    def test_trial_metadata_reconciles_through_v1_and_v2(
        self,
        configured_pytester: Pytester,
        dist_mode: str,
    ) -> None:
        _configure_shadow_snapshot(configured_pytester)
        configured_pytester.makepyfile(
            test_shadow_trial="""\
            import pytest

            from rampart import (
                BaseExecution,
                Result,
                SafetyStatus,
                execute_trials_async,
                record_result,
            )


            class Execution(BaseExecution):
                def __init__(self, *, index):
                    super().__init__()
                    self._index = index

                @property
                def strategy_name(self):
                    return "test"

                async def _execute_async(self, *, adapter):
                    return Result(
                        status=SafetyStatus.SAFE,
                        summary=f"trial-{self._index}",
                    )


            @pytest.mark.xdist_group(name="shadow-trial")
            @pytest.mark.harm("test")
            async def test_shadow_trial():
                record_result(Result(status=SafetyStatus.SAFE, summary="plain"))
                index = 0

                def factory():
                    nonlocal index
                    execution = Execution(index=index)
                    index += 1
                    return execution

                batch = await execute_trials_async(
                    execution_factory=factory,
                    adapter=object(),
                    count=2,
                )
                assert batch
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            f"--dist={dist_mode}",
        )

        expected_items = 2 if dist_mode == "each" else 1
        expected_results = expected_items * 3
        result.assert_outcomes(passed=expected_items)
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        report = reports[0]
        assert report["total_runs"] == expected_results
        assert len(report["trial_batches"]) == expected_items
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["faults"] == []
        assert snapshot["incomplete"] is False
        assert snapshot["v1_results"] == expected_results
        assert snapshot["shadow_results"] == expected_results
        v2_records = snapshot["v2_records"]
        assert snapshot["v1_records"] == [
            {key: value for key, value in record.items() if key != "slot_index"}
            for record in v2_records
        ]
        assert all(
            record["slot_index"] == record["result_index"] for record in v2_records
        )
        by_worker: dict[str, list[dict[str, Any]]] = {}
        for record in v2_records:
            by_worker.setdefault(record["worker_id"], []).append(record)
        assert all(
            [record["slot_index"] for record in records] == [0, 1, 2]
            for records in by_worker.values()
        )
        for records in by_worker.values():
            by_summary = {record["summary"]: record for record in records}
            assert by_summary["plain"]["trial_metadata"] == {}
            for index in range(2):
                metadata = by_summary[f"trial-{index}"]["trial_metadata"]
                assert metadata["_rampart_trial_batch_index"] == index
                assert metadata["_rampart_trial_batch_count"] == 2
                assert metadata["_rampart_trial_batch_schema"] == (
                    "rampart.trial-batch.v1"
                )
        batch_ids = [summary["batch_id"] for summary in report["trial_batches"]]
        first_seen = list(
            dict.fromkeys(
                item["metadata"]["_rampart_trial_batch_id"]
                for item in _flatten_results(report)
                if "_rampart_trial_batch_id" in item["metadata"]
            ),
        )
        assert batch_ids == first_seen


class TestXdistShadowLifecycle:
    @pytest.mark.parametrize("dist_mode", ["load", "loadgroup", "each"])
    def test_v1_v2_multisets_match_under_scheduler(
        self,
        configured_pytester: Pytester,
        dist_mode: str,
    ) -> None:
        _configure_shadow_snapshot(configured_pytester)
        _make_shadow_result_tests(configured_pytester)

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            f"--dist={dist_mode}",
        )

        expected = 4 if dist_mode == "each" else 2
        result.assert_outcomes(passed=expected)
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["v1_results"] == expected
        assert snapshot["shadow_results"] == expected
        assert snapshot["faults"] == []
        assert snapshot["incomplete"] is False

    def test_same_worker_attempts_have_distinct_sequences(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _configure_shadow_snapshot(configured_pytester)
        _make_shadow_result_tests(configured_pytester)

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
        )

        result.assert_outcomes(passed=2)
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["sequences"] == [[1, 2]]
        assert snapshot["delivery_count"] == 2
        assert sorted(snapshot["summaries"]) == ["one", "two"]

    def test_one_item_preserves_multiple_result_order(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _configure_shadow_snapshot(configured_pytester)
        configured_pytester.makepyfile(
            test_multiple="""\
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus


            def test_multiple_results():
                record_result(Result(status=SafetyStatus.SAFE, summary="first"))
                record_result(Result(status=SafetyStatus.SAFE, summary="second"))
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
        )

        result.assert_outcomes(passed=1)
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["slot_indexes"] == [[0, 1]]
        assert snapshot["summaries"] == ["first", "second"]
        assert snapshot["v1_results"] == snapshot["shadow_results"] == 2

    def test_non_text_payload_reconciles_cleanly(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _configure_shadow_snapshot(configured_pytester)
        configured_pytester.makepyfile(
            test_non_text="""\
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus
            from rampart.core.types import (
                Payload,
                PayloadFormat,
                Request,
                Response,
                Turn,
            )


            def test_non_text_payload():
                record_result(
                    Result(
                        status=SafetyStatus.SAFE,
                        summary="html payload",
                        turns=[
                            Turn(
                                request=Request(
                                    attachments=[
                                        Payload(
                                            content="<b>payload</b>",
                                            format=PayloadFormat.HTML,
                                        )
                                    ]
                                ),
                                response=Response(text="ok"),
                            )
                        ],
                    )
                )
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
        )

        result.assert_outcomes(passed=1)
        assert result.ret == pytest.ExitCode.OK
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["v1_results"] == snapshot["shadow_results"] == 1
        assert snapshot["faults"] == []

    @pytest.mark.parametrize("manifest_mode", ["missing", "empty"])
    def test_invalid_clean_manifest_is_incomplete_nonzero(
        self,
        configured_pytester: Pytester,
        manifest_mode: str,
    ) -> None:
        _configure_shadow_snapshot(
            configured_pytester,
            manifest_mode=manifest_mode,
        )
        configured_pytester.makepyfile(
            test_manifest="""\
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus


            def test_manifest():
                record_result(Result(status=SafetyStatus.SAFE, summary="kept by v1"))
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
        )

        result.assert_outcomes(passed=1)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["v1_results"] == snapshot["shadow_results"] == 1
        assert snapshot["incomplete"] is True
        assert any(
            code.startswith(("missing-manifest:", "malformed-manifest:"))
            for code in snapshot["faults"]
        )

    def test_malformed_report_attribute_is_incomplete_nonzero(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _configure_shadow_snapshot(
            configured_pytester,
            report_mode="malformed",
        )
        configured_pytester.makepyfile(
            test_malformed="""\
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus


            def test_malformed():
                record_result(Result(status=SafetyStatus.SAFE, summary="v1 survives"))
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
        )

        result.assert_outcomes(passed=1)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["v1_results"] == 1
        assert snapshot["shadow_results"] == 0
        assert snapshot["incomplete"] is True
        assert any(
            code.startswith("malformed-envelope:") for code in snapshot["faults"]
        )

    def test_worker_loss_retains_delivered_shadow_only(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _configure_shadow_snapshot(configured_pytester)
        configured_pytester.makepyfile(
            test_worker_loss="""\
            import os

            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus


            def test_a_delivered():
                record_result(Result(status=SafetyStatus.SAFE, summary="delivered"))


            def test_b_worker_loss():
                os._exit(3)
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
            "--max-worker-restart=0",
        )

        assert result.ret != pytest.ExitCode.OK
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["summaries"] == ["delivered"]
        assert snapshot["shadow_results"] == 1
        assert snapshot["v1_results"] == 0
        assert snapshot["incomplete"] is True

    def test_size_drop_keeps_shadow_siblings_and_is_nonzero(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _configure_shadow_snapshot(configured_pytester)
        configured_pytester.makepyfile(
            test_drop="""\
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus


            def test_drop():
                record_result(Result(status=SafetyStatus.SAFE, summary="first"))
                record_result(Result(status=SafetyStatus.SAFE, summary="x" * 5000))
                record_result(Result(status=SafetyStatus.SAFE, summary="third"))
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
            "--rampart-xdist-max-bytes=1000",
        )

        result.assert_outcomes(passed=1)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        snapshot = _load_shadow_snapshot(configured_pytester)
        assert snapshot["summaries"] == ["first", "third"]
        assert snapshot["shadow_results"] == 2
        assert snapshot["incomplete"] is True

    def test_private_envelope_does_not_leak_to_junit(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_junit_shadow="""\
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus


            def test_junit_shadow():
                record_result(
                    Result(
                        status=SafetyStatus.SAFE,
                        summary="shadow-junit-secret",
                    )
                )
            """,
        )
        xml_path = configured_pytester.path / "shadow-report.xml"

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "1",
            f"--junitxml={xml_path}",
        )

        result.assert_outcomes(passed=1)
        assert result.ret == pytest.ExitCode.OK
        xml_text = xml_path.read_text(encoding="utf-8")
        assert REPORT_ATTRIBUTE not in xml_text
        assert "shadow-junit-secret" not in xml_text


class TestTrialMarkerDeprecation:
    @pytest.mark.parametrize(
        "args",
        [(), ("-n", "2")],
        ids=["serial", "xdist"],
    )
    def test_warning_is_visible_once(
        self,
        configured_pytester: Pytester,
        args: tuple[str, ...],
    ) -> None:
        configured_pytester.makepyfile(
            test_deprecated_trial="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus

            @pytest.mark.trial(n=2)
            def test_deprecated_trial():
                record_result(Result(status=SafetyStatus.SAFE, summary="safe"))
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            *args,
        )

        result.assert_outcomes(passed=2, warnings=2)
        warning_lines = [
            line
            for line in result.outlines
            if "clone-based @pytest.mark.trial marker is deprecated" in line
        ]
        assert len(warning_lines) == 1
        assert _EXPECTED_TRIAL_MARKER_DEPRECATION_MESSAGE in warning_lines[0]
        assert any("PytestDeprecationWarning" in line for line in warning_lines)

    def test_warning_error_is_contained_after_report(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_warning_error="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus

            @pytest.mark.trial(n=2)
            def test_warning_error():
                record_result(Result(status=SafetyStatus.SAFE, summary="safe"))
            """,
        )

        result = configured_pytester.runpytest_subprocess(
            "-p",
            "no:cacheprovider",
            "-W",
            "error::pytest.PytestDeprecationWarning",
        )

        result.assert_outcomes(passed=2, warnings=2)
        assert result.ret == pytest.ExitCode.OK
        assert "RAMPART Safety Summary" in "\n".join(result.outlines)
        warning_lines = [
            line
            for line in result.outlines
            if "clone-based @pytest.mark.trial marker is deprecated" in line
        ]
        assert len(warning_lines) == 1
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        assert reports[0]["total_runs"] == 2

    def test_warning_ignore_filter_is_honored(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_warning_ignore="""
            import pytest
            from rampart import record_result
            from rampart.core.result import Result, SafetyStatus

            @pytest.mark.trial(n=2)
            def test_warning_ignore():
                record_result(Result(status=SafetyStatus.SAFE, summary="safe"))
            """,
        )

        result = configured_pytester.runpytest_subprocess(
            "-p",
            "no:cacheprovider",
            "-W",
            "ignore::pytest.PytestDeprecationWarning",
        )

        result.assert_outcomes(passed=2, warnings=1)
        assert result.ret == pytest.ExitCode.OK
        assert "clone-based @pytest.mark.trial marker is deprecated" not in "\n".join(
            result.outlines
        )


class TestXdistMetadata:
    def test_report_includes_xdist_metadata(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _setup_simple_tests(configured_pytester)
        configured_pytester.runpytest("-p", "no:cacheprovider", "-n", "2")
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        metadata = reports[0].get("metadata", {})
        assert metadata.get("xdist_active") is True
        assert metadata.get("worker_count") == 2
        assert "dist_mode" in metadata
        assert "population_summary" in reports[0]

    def test_size_cap_marks_run_incomplete(self, configured_pytester: Pytester) -> None:
        """Forcing a 1-byte cap surfaces incompleteness in report metadata.

        Triggers the truncation path so the controller must record
        ``incomplete=True`` plus a reason in the merged report.
        """
        _setup_simple_tests(configured_pytester)
        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-n",
            "2",
            "--rampart-xdist-max-bytes=1",
        )
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        reports = _load_reports(configured_pytester)
        assert len(reports) == 1
        metadata = reports[0].get("metadata", {})
        assert metadata.get("incomplete") is True
        reasons = metadata.get("incomplete_reasons", [])
        assert any("truncated" in r for r in reasons)


class TestCollectOnly:
    def test_collect_only_does_not_emit_reports(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _setup_simple_tests(configured_pytester)
        configured_pytester.runpytest("-p", "no:cacheprovider", "--collect-only")
        # No sinks emit when no tests run
        marker = configured_pytester.path / "rampart_report_dir.txt"
        if marker.exists():
            out_dir = Path(marker.read_text().strip())
            if out_dir.exists():
                reports = list(out_dir.glob("run_report_*.json"))
                assert reports == []

    def test_collect_only_trial_skips_failed_gate(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_collect_trial="""
            import pytest

            @pytest.mark.trial(n=2)
            def test_no_result():
                pass
            """,
        )

        result = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "--collect-only",
        )

        assert result.ret == pytest.ExitCode.OK
        assert "forcing non-zero exit status" not in "\n".join(result.outlines)


class TestCloneIdDeterminism:
    def test_trial_clone_ids_deterministic_across_processes(
        self,
        configured_pytester: Pytester,
    ) -> None:
        configured_pytester.makepyfile(
            test_det="""
            import pytest

            @pytest.mark.trial(n=3)
            def test_x():
                pass
            """,
        )
        result_serial: RunResult = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
        )
        result_parallel: RunResult = configured_pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            "-n",
            "2",
        )
        assert result_serial.ret == pytest.ExitCode.OK
        assert result_parallel.ret == pytest.ExitCode.OK

        def _trial_ids(lines: list[str]) -> list[str]:
            return sorted(line.strip() for line in lines if "trial-" in line)

        serial_ids = _trial_ids(result_serial.outlines)
        parallel_ids = _trial_ids(result_parallel.outlines)
        # Under xdist --collect-only, both should produce the same
        # deterministic clone IDs so that workers can match them.
        if serial_ids and parallel_ids:
            assert serial_ids == parallel_ids


class TestSinkFixtureDeprecation:
    """End-to-end deprecation-warning contract for the ``rampart_sinks`` fixture.

    The fixture warns wherever it is resolved: single-process and on the xdist
    controller. The list form's silence is covered by the fast unit tests in
    ``test_xdist.py::TestSinkDeprecationWarning``.
    """

    _DEPRECATION_LINE = "*rampart_sinks fixture is deprecated*"

    def test_single_process_fixture_warns(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _setup_simple_tests(configured_pytester)
        result = configured_pytester.runpytest("-p", "no:cacheprovider")
        result.assert_outcomes(passed=4)
        result.stdout.fnmatch_lines([self._DEPRECATION_LINE])

    def test_controller_fixture_warns_under_xdist(
        self,
        configured_pytester: Pytester,
    ) -> None:
        _setup_simple_tests(configured_pytester)
        result = configured_pytester.runpytest("-p", "no:cacheprovider", "-n", "2")
        result.assert_outcomes(passed=4)
        result.stdout.fnmatch_lines([self._DEPRECATION_LINE])
