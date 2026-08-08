# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the RAMPART pytest plugin hooks."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Never, cast
from unittest.mock import MagicMock

import pytest

from rampart.core.result import Result, SafetyStatus
from rampart.core.types import ObservabilityLevel
from rampart.pytest_plugin._collection import (
    ResultCollectionHandler,
    ResultCollector,
    _active_collector,
    activate_collector,
    deactivate_collector,
)
from rampart.pytest_plugin._session import RampartSession
from rampart.pytest_plugin.plugin import (
    _absorb_results,
    _call_results_key,
    _emit_sinks,
    _enforce_incomplete_exit_status,
    _enforce_trial_gate_exit_status,
    _evaluate_gates,
    _has_sink_hook_impl,
    _resolve_hook_sinks,
    _resolve_trial_n,
    _resolve_trial_threshold,
    _warn_trial_marker_deprecated,
    _write_result_line,
    _write_trial_group_lines,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_makereport,
    pytest_sessionfinish,
    pytest_terminal_summary,
    pytest_unconfigure,
)
from rampart.reporting.sink import ReportSink, TestRunReport

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter


class _HostileRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr failed")

    def __str__(self) -> str:
        raise RuntimeError("str failed")

    def __iter__(self) -> Never:
        raise RuntimeError("iter failed")

    def __len__(self) -> int:
        raise RuntimeError("len failed")


class _HostileList(list):  # ruff: ignore[subclass-builtin]
    def __iter__(self) -> Never:
        raise RuntimeError("iter failed")


class _HostileMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name == "__name__":
            raise RuntimeError("name failed")
        return type.__getattribute__(
            cls,
            name,
        )


class _HostileTypeName(metaclass=_HostileMeta):
    pass


class _DescriptorMeta(type):
    @property
    def __name__(cls) -> str:  # ruff: ignore[bad-dunder-method-name]
        del cls
        raise SystemExit("name descriptor invoked")


class _DescriptorTypeName(metaclass=_DescriptorMeta):
    pass


def _raise_repr(value: object) -> str:
    del value
    raise RuntimeError("repr failed")


def _assert_no_terminal_controls(value: str) -> None:
    assert all(
        ord(character) >= 0x20
        and ord(character) != 0x7F
        and not 0x80 <= ord(character) <= 0x9F
        for character in value
    )


class _StashStub:
    """Minimal pytest.Stash test double backed by a dict."""

    def __init__(self) -> None:
        self._data: dict[Any, Any] = {}

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data[key] = value

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def __delitem__(self, key: Any) -> None:
        del self._data[key]

    def get(self, key: Any, default: Any = None) -> Any:
        """Return value for key, or default."""
        return self._data.get(key, default)

    def pop(self, key: Any, *args: Any) -> Any:
        """Remove and return value for key."""
        return self._data.pop(key, *args)


class _ConfigStub:
    """Minimal pytest.Config test double with stash support."""

    def __init__(self) -> None:
        self._ini_lines: list[tuple[str, str]] = []
        self.stash = _StashStub()

    def addinivalue_line(self, name: str, line: str) -> None:
        """Record marker registrations."""
        self._ini_lines.append((name, line))


def _format_record(record: logging.LogRecord) -> str:
    return logging.Formatter("%(levelname)s:%(message)s").format(record)


class TestDefaultHandlerFactory:
    """Handler factory is set during configure and cleared on unconfigure."""

    def test_configure_sets_factory(self) -> None:
        config: Any = _ConfigStub()
        pytest_configure(config)
        try:
            from rampart.core.execution import _default_handler_factory

            handlers = _default_handler_factory()
            assert len(handlers) == 1
            assert isinstance(handlers[0], ResultCollectionHandler)
        finally:
            pytest_unconfigure(config)

    def test_unconfigure_clears_factory(self) -> None:
        config: Any = _ConfigStub()
        pytest_configure(config)
        pytest_unconfigure(config)

        from rampart.core.execution import _default_handler_factory

        assert _default_handler_factory() == []

    def test_configure_creates_session_in_stash(self) -> None:
        config: Any = _ConfigStub()
        pytest_configure(config)
        try:
            from rampart.pytest_plugin.plugin import _rampart_key

            assert isinstance(config.stash.get(_rampart_key), RampartSession)
        finally:
            pytest_unconfigure(config)

    def test_unconfigure_removes_session_from_stash(self) -> None:
        config: Any = _ConfigStub()
        pytest_configure(config)
        pytest_unconfigure(config)

        from rampart.pytest_plugin.plugin import _rampart_key

        assert config.stash.get(_rampart_key) is None


class TestRampartSession:
    """RampartSession accumulates results and builds reports."""

    def test_absorb_accumulates_results(self) -> None:
        session = RampartSession()
        collector = ResultCollector()
        collector.record(
            result=Result(status=SafetyStatus.SAFE, summary="ok"),
        )
        node = MagicMock()
        node.nodeid = "test_file.py::test_absorb"

        session.absorb(node=node, collector=collector)

        assert session.has_results
        report = session.build_report()
        assert report.total_runs == 1
        assert report.passed == 1

    def test_absorb_retains_same_node_attempts_in_order(self) -> None:
        session = RampartSession()
        node = MagicMock()
        node.nodeid = "test_file.py::test_repeat"
        node.get_closest_marker.return_value = None
        first_collector = ResultCollector()
        first_original = Result(status=SafetyStatus.SAFE, summary="first")
        first_collector.record(result=first_original)
        second_collector = ResultCollector()
        second_original = Result(status=SafetyStatus.ERROR, summary="second")
        second_collector.record(result=second_original)

        first = session.absorb(node=node, collector=first_collector)
        second = session.absorb(node=node, collector=second_collector)

        grouped = session.results_by_nodeid[node.nodeid]
        report = session.build_report()
        assert [result.summary for result in grouped] == ["first", "second"]
        assert [result.summary for result in report.results] == ["first", "second"]
        assert [result.metadata["_rampart_result_index"] for result in grouped] == [
            0,
            1,
        ]
        assert first == (grouped[0],)
        assert second == (grouped[1],)
        assert first[0] is grouped[0]
        assert second[0] is grouped[1]
        assert first[0] is not first_original
        assert second[0] is not second_original
        assert report.total_runs == 2
        assert report.population_summary().total_runs == 2

    def test_has_results_false_when_empty(self) -> None:
        session = RampartSession()
        assert not session.has_results

    def test_absorb_failure_log_escapes_complete_traceback(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        raw_error = "absorb\x1b\t\n\r\x7f\x9b"
        raw_nodeid = "test\x1b\nnode\x9b"
        error = RuntimeError(raw_error)
        session = MagicMock()
        session.absorb.side_effect = error
        node = MagicMock()
        node.nodeid = raw_nodeid

        with caplog.at_level(logging.WARNING):
            _absorb_results(
                rampart_session=session,
                node=node,
                collector=ResultCollector(),
            )

        formatted = _format_record(caplog.records[-1])
        assert r"test\x1b\x0anode\x9b" in formatted
        assert r"RuntimeError: absorb\x1b\x09\x0a\x0d\x7f\x9b" in formatted
        assert "Traceback (most recent call last):" in formatted
        assert "\x1b" not in formatted
        assert "\t" not in formatted
        assert "\n" not in formatted
        assert "\r" not in formatted
        assert "\x7f" not in formatted
        assert "\x9b" not in formatted
        assert str(error) == raw_error

    def test_build_report_counts(self) -> None:
        session = RampartSession()

        collector = ResultCollector()
        collector.record(
            result=Result(status=SafetyStatus.SAFE, summary="s"),
        )
        collector.record(
            result=Result(status=SafetyStatus.UNSAFE, summary="u"),
        )
        collector.record(
            result=Result(status=SafetyStatus.ERROR, summary="e"),
        )
        node = MagicMock()
        node.nodeid = "test_file.py::test_counts"

        session.absorb(node=node, collector=collector)
        report = session.build_report()

        assert report.total_runs == 3
        assert report.passed == 1
        assert report.failed == 1
        assert report.errors == 1

    def test_record_trial_group(self) -> None:
        session = RampartSession()

        items: list[Any] = [MagicMock() for _ in range(5)]
        statuses = [
            SafetyStatus.UNSAFE,
            SafetyStatus.SAFE,
            SafetyStatus.UNSAFE,
            SafetyStatus.ERROR,
            SafetyStatus.SAFE,
        ]
        for idx, item in enumerate(items):
            item.nodeid = f"test_file.py::test_example[trial-{idx}]"
            collector = ResultCollector()
            collector.record(
                result=Result(
                    status=statuses[idx],
                    summary=f"trial-{idx}",
                ),
            )
            session.absorb(node=item, collector=collector)

        session.record_trial_group(
            base_nodeid="test_example",
            clone_nodeids=[item.nodeid for item in items],
            threshold=0.3,
        )

        groups = session.trial_groups
        assert "test_example" in groups
        group = groups["test_example"]
        assert group.total == 5
        assert group.safe == 2
        assert group.unsafe == 2
        assert group.errors == 1
        assert group.threshold == pytest.approx(0.3)
        assert group.pass_rate == pytest.approx(0.4)
        assert not group.passed  # UNSAFE present → always fails

    def test_record_trial_group_all_errors(self) -> None:
        session = RampartSession()

        items: list[Any] = [MagicMock() for _ in range(3)]
        for idx, item in enumerate(items):
            item.nodeid = f"test_file.py::test_err[trial-{idx}]"
            collector = ResultCollector()
            collector.record(
                result=Result(
                    status=SafetyStatus.ERROR,
                    summary=f"err-{idx}",
                ),
            )
            session.absorb(node=item, collector=collector)

        session.record_trial_group(
            base_nodeid="test_err",
            clone_nodeids=[item.nodeid for item in items],
            threshold=1.0,
        )

        group = session.trial_groups["test_err"]
        assert group.errors == 3
        assert group.unsafe == 0
        assert group.pass_rate == pytest.approx(0.0)
        assert not group.passed

    def test_record_trial_group_all_no_result_fails(self) -> None:
        session = RampartSession()
        clone_nodeids = [
            "test_file.py::test_missing[trial-0]",
            "test_file.py::test_missing[trial-1]",
        ]

        session.record_trial_group(
            base_nodeid="test_missing",
            clone_nodeids=clone_nodeids,
            threshold=1.0,
        )

        group = session.trial_groups["test_missing"]
        assert group.total == 2
        assert group.no_result == 2
        assert group.pass_rate == pytest.approx(0.0)
        assert not group.passed

    def test_record_trial_group_fails_below_threshold(self) -> None:
        session = RampartSession()

        items: list[Any] = [MagicMock() for _ in range(4)]
        statuses = [
            SafetyStatus.SAFE,
            SafetyStatus.SAFE,
            SafetyStatus.UNDETERMINED,
            SafetyStatus.UNDETERMINED,
        ]
        for idx, item in enumerate(items):
            item.nodeid = f"test_file.py::test_thresh[trial-{idx}]"
            collector = ResultCollector()
            collector.record(
                result=Result(
                    status=statuses[idx],
                    summary=f"trial-{idx}",
                ),
            )
            session.absorb(node=item, collector=collector)

        session.record_trial_group(
            base_nodeid="test_thresh",
            clone_nodeids=[item.nodeid for item in items],
            threshold=0.75,
        )

        group = session.trial_groups["test_thresh"]
        assert group.unsafe == 0
        assert group.safe == 2
        assert group.pass_rate == pytest.approx(0.5)
        assert not group.passed  # no UNSAFE, but pass rate below threshold

    def test_record_trial_group_passes_when_all_safe(self) -> None:
        session = RampartSession()

        items: list[Any] = [MagicMock() for _ in range(3)]
        for idx, item in enumerate(items):
            item.nodeid = f"test_file.py::test_all_safe[trial-{idx}]"
            collector = ResultCollector()
            collector.record(
                result=Result(
                    status=SafetyStatus.SAFE,
                    summary=f"trial-{idx}",
                ),
            )
            session.absorb(node=item, collector=collector)

        session.record_trial_group(
            base_nodeid="test_all_safe",
            clone_nodeids=[item.nodeid for item in items],
            threshold=0.5,
        )

        group = session.trial_groups["test_all_safe"]
        assert group.unsafe == 0
        assert group.safe == 3
        assert group.pass_rate == pytest.approx(1.0)
        assert group.passed  # all SAFE and at/above threshold

    def test_trial_group_counts_clones_not_same_node_attempts(self) -> None:
        session = RampartSession()
        repeated = MagicMock()
        repeated.nodeid = "test_file.py::test_repeat[trial-0]"
        repeated.get_closest_marker.return_value = None
        other = MagicMock()
        other.nodeid = "test_file.py::test_repeat[trial-1]"
        other.get_closest_marker.return_value = None
        for status in (SafetyStatus.SAFE, SafetyStatus.UNSAFE):
            collector = ResultCollector()
            collector.record(result=Result(status=status, summary=status.value))
            session.absorb(node=repeated, collector=collector)
        collector = ResultCollector()
        collector.record(result=Result(status=SafetyStatus.SAFE, summary="safe"))
        session.absorb(node=other, collector=collector)

        session.record_trial_group(
            base_nodeid="test_file.py::test_repeat",
            clone_nodeids=[repeated.nodeid, other.nodeid],
            threshold=0.5,
        )

        group = session.trial_groups["test_file.py::test_repeat"]
        assert group.total == 2
        assert group.safe == 1
        assert group.unsafe == 1
        assert not group.passed
        assert session.build_report().total_runs == 3

    def test_record_trial_group_empty_items_noop(self) -> None:
        session = RampartSession()
        session.record_trial_group(
            base_nodeid="test_empty",
            clone_nodeids=[],
            threshold=1.0,
        )
        assert "test_empty" not in session.trial_groups


def _make_trial_item(
    *,
    n: int = 3,
    threshold: float = 1.0,
    nodeid: str = "test_file.py::test_example",
    name: str = "test_example",
) -> MagicMock:
    """Build a mock pytest.Item with a trial marker."""
    marker = pytest.mark.trial(n=n, threshold=threshold).mark
    item = MagicMock()
    item.get_closest_marker.return_value = marker
    item.nodeid = nodeid
    item.name = name
    item.originalname = name
    item.parent = MagicMock()
    item.function = lambda: None
    return item


def _make_plain_item(
    *,
    nodeid: str = "test_file.py::test_plain",
    name: str = "test_plain",
) -> MagicMock:
    """Build a mock pytest.Item without a trial marker."""
    item = MagicMock()
    item.get_closest_marker.return_value = None
    item.nodeid = nodeid
    item.originalname = name
    return item


class TestTrialCloning:
    """Trial cloning produces n items with distinct [trial-N] node ids."""

    def test_trial_cloning_produces_n_items(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        item = _make_trial_item(n=3)
        clone_instances = [MagicMock() for _ in range(3)]
        for clone in clone_instances:
            clone.iter_markers.return_value = []
        mock_from_parent = MagicMock(side_effect=clone_instances)
        # type(item).from_parent is used in plugin, so patch it on the mock's type
        type(item).from_parent = mock_from_parent

        items: list[Any] = [item]
        config = MagicMock()
        pytest_collection_modifyitems(
            config=cast("pytest.Config", config),
            items=items,
        )

        assert len(items) == 3
        calls = mock_from_parent.call_args_list
        for i, call in enumerate(calls):
            assert call.kwargs["name"] == f"test_example[trial-{i}]"

    def test_trial_n_zero_raises_usage_error(self) -> None:
        item = _make_trial_item(n=0)
        items: list[Any] = [item]
        config = MagicMock()

        with pytest.raises(pytest.UsageError, match="must be >= 1"):
            pytest_collection_modifyitems(
                config=cast("pytest.Config", config),
                items=items,
            )

    def test_non_trial_items_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plain = _make_plain_item()
        trial = _make_trial_item(n=2)
        clone_instances = [MagicMock() for _ in range(2)]
        for clone in clone_instances:
            clone.iter_markers.return_value = []
        type(trial).from_parent = MagicMock(side_effect=clone_instances)

        items: list[Any] = [plain, trial]
        config = MagicMock()
        pytest_collection_modifyitems(
            config=cast("pytest.Config", config),
            items=items,
        )

        assert items[0] is plain
        assert len(items) == 3

    def test_trial_item_with_no_parent_raises(self) -> None:
        item = _make_trial_item(n=2)
        item.parent = None

        items: list[Any] = [item]
        config = MagicMock()

        with pytest.raises(pytest.UsageError, match="no parent"):
            pytest_collection_modifyitems(
                config=cast("pytest.Config", config),
                items=items,
            )


class TestResolveTrialN:
    """_resolve_trial_n extracts n from positional and keyword args."""

    def test_keyword_n(self) -> None:
        marker = pytest.mark.trial(n=7).mark
        assert _resolve_trial_n(marker) == 7

    def test_positional_n(self) -> None:
        marker = pytest.mark.trial(5).mark
        assert _resolve_trial_n(marker) == 5

    def test_keyword_takes_precedence(self) -> None:
        marker = pytest.mark.trial(3, n=10).mark
        assert _resolve_trial_n(marker) == 10

    def test_defaults_to_one(self) -> None:
        marker = pytest.mark.trial(threshold=0.5).mark
        assert _resolve_trial_n(marker) == 1

    def test_string_n_raises_usage_error(self) -> None:
        """Non-integer n raises UsageError instead of a confusing TypeError."""
        marker = pytest.mark.trial(n="five").mark
        with pytest.raises(pytest.UsageError, match="must be an integer"):
            _resolve_trial_n(marker)

    def test_positional_string_raises_usage_error(self) -> None:
        """Non-integer positional arg raises UsageError."""
        marker = pytest.mark.trial("hello").mark
        with pytest.raises(pytest.UsageError, match="must be an integer"):
            _resolve_trial_n(marker)

    def test_float_n_raises_usage_error(self) -> None:
        """Float n raises UsageError."""
        marker = pytest.mark.trial(n=3.5).mark
        with pytest.raises(pytest.UsageError, match="must be an integer"):
            _resolve_trial_n(marker)

    def test_bool_n_raises_usage_error(self) -> None:
        """Bool n raises UsageError (bool is subclass of int)."""
        marker = pytest.mark.trial(n=True).mark
        with pytest.raises(pytest.UsageError, match="must be an integer"):
            _resolve_trial_n(marker)

    def test_bool_false_raises_usage_error(self) -> None:
        """False also rejected despite bool being int subclass."""
        marker = pytest.mark.trial(n=False).mark
        with pytest.raises(pytest.UsageError, match="must be an integer"):
            _resolve_trial_n(marker)

    def test_hostile_repr_raises_usage_error(self) -> None:
        marker = pytest.mark.trial(n=_HostileRepr()).mark
        with pytest.raises(pytest.UsageError, match="must be an integer"):
            _resolve_trial_n(marker)

    def test_huge_negative_integer_raises_usage_error(self) -> None:
        marker = pytest.mark.trial(n=-(10**5000)).mark
        with pytest.raises(pytest.UsageError, match="must be >= 1") as exc_info:
            _resolve_trial_n(marker)
        assert len(str(exc_info.value)) < 200


class TestResolveTrialThreshold:
    def test_defaults_to_one(self) -> None:
        marker = pytest.mark.trial(n=3).mark
        assert _resolve_trial_threshold(marker) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.25, 0.25), (1, 1.0), ("0.75", 0.75)],
    )
    def test_accepts_finite_values_in_range(
        self,
        value: float | str,
        expected: float,
    ) -> None:
        marker = pytest.mark.trial(threshold=value).mark
        assert _resolve_trial_threshold(marker) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value",
        [
            0.0,
            -0.1,
            1.1,
            math.nan,
            math.inf,
            -math.inf,
            True,
            False,
            "not-a-number",
            None,
            10**400,
        ],
    )
    def test_rejects_invalid_values(self, value: object) -> None:
        marker = pytest.mark.trial(threshold=value).mark
        with pytest.raises(pytest.UsageError, match=r"finite number in \(0, 1\]"):
            _resolve_trial_threshold(marker)

    def test_huge_integer_raises_bounded_usage_error(self) -> None:
        marker = pytest.mark.trial(threshold=10**5000).mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert "finite number in (0, 1]" in str(exc_info.value)
        assert len(str(exc_info.value)) < 200

    def test_hostile_repr_raises_usage_error(self) -> None:
        marker = pytest.mark.trial(threshold=_HostileRepr()).mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert "finite number in (0, 1]" in str(exc_info.value)
        assert len(str(exc_info.value)) < 200

    def test_exact_container_diagnostic_does_not_traverse(self) -> None:
        marker = pytest.mark.trial(threshold=[_HostileRepr()]).mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert "<list len=1>" in str(exc_info.value)

    def test_nested_huge_integer_diagnostic_is_bounded(self) -> None:
        value: object = 10**5000
        for _ in range(6):
            value = [value] * 6
        marker = pytest.mark.trial(threshold=value).mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert len(str(exc_info.value)) < 200

    def test_long_type_name_diagnostic_is_bounded(self) -> None:
        hostile_type = type("X" * 10_000, (), {"__repr__": _raise_repr})
        marker = pytest.mark.trial(threshold=hostile_type()).mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert len(str(exc_info.value)) < 200

    def test_hostile_metaclass_name_hook_is_not_invoked(self) -> None:
        marker = pytest.mark.trial(threshold=_HostileTypeName()).mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert len(str(exc_info.value)) < 200

    def test_metaclass_name_descriptor_is_not_invoked(self) -> None:
        marker = pytest.mark.trial(threshold=_DescriptorTypeName()).mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert "<object>" in str(exc_info.value)

    def test_value_and_type_name_controls_are_escaped(self) -> None:
        controlled_type = type("Bad\r\n\x1b]0;name\x07\x80", (), {})
        values = [
            "\r\n\x1b[31m\x1b]0;value\x07\x80",
            controlled_type(),
        ]
        for value in values:
            marker = pytest.mark.trial(threshold=value).mark
            with pytest.raises(pytest.UsageError) as exc_info:
                _resolve_trial_threshold(marker)
            _assert_no_terminal_controls(str(exc_info.value))

    def test_ordinary_invalid_value_remains_helpful(self) -> None:
        marker = pytest.mark.trial(threshold="not-a-number").mark
        with pytest.raises(pytest.UsageError) as exc_info:
            _resolve_trial_threshold(marker)
        assert "'not-a-number'" in str(exc_info.value)


class TestWriteResultLine:
    """_write_result_line writes formatted status, summary, and observability level."""

    def test_safe_result_includes_observability(self) -> None:
        reporter = MagicMock()
        result = Result(
            status=SafetyStatus.SAFE,
            summary="ok",
            observability_level=ObservabilityLevel.RESPONSE_ONLY,
        )
        _write_result_line(
            terminalreporter=cast("TerminalReporter", reporter),
            result=result,
        )
        reporter.write_line.assert_called_once_with("  PASS  ok (response_only)")

    def test_unsafe_result_includes_observability(self) -> None:
        reporter = MagicMock()
        result = Result(
            status=SafetyStatus.UNSAFE,
            summary="bad",
            observability_level=ObservabilityLevel.TOOL_AND_SIDE_EFFECTS,
        )
        _write_result_line(
            terminalreporter=cast("TerminalReporter", reporter),
            result=result,
        )
        reporter.write_line.assert_called_once_with(
            "  FAIL  bad (tool_and_side_effects)",
        )

    def test_with_test_name(self) -> None:
        reporter = MagicMock()
        result = Result(
            status=SafetyStatus.SAFE,
            summary="SAFE",
            observability_level=ObservabilityLevel.TOOL_ONLY,
        )
        _write_result_line(
            terminalreporter=cast("TerminalReporter", reporter),
            result=result,
            test_name="test_exfil",
        )
        reporter.write_line.assert_called_once_with(
            "  PASS  test_exfil -- SAFE (tool_only)",
        )

    def test_controls_escaped_in_summary(self) -> None:
        reporter = MagicMock()
        result = Result(
            status=SafetyStatus.SAFE,
            summary="\x1b[31mevil\x1b[0m\tline\nnext\r\x7f\x9b",
        )
        _write_result_line(
            terminalreporter=cast("TerminalReporter", reporter),
            result=result,
        )
        line = reporter.write_line.call_args[0][0]
        assert r"\x1b[31mevil\x1b[0m\x09line\x0anext\x0d\x7f\x9b" in line
        assert "\x1b" not in line
        assert "\n" not in line

    def test_controls_escaped_in_test_name(self) -> None:
        reporter = MagicMock()
        result = Result(status=SafetyStatus.SAFE, summary="ok")
        _write_result_line(
            terminalreporter=cast("TerminalReporter", reporter),
            result=result,
            test_name="test\x1b\nname\x9b",
        )
        line = reporter.write_line.call_args[0][0]
        assert r"test\x1b\x0aname\x9b" in line
        assert "\x1b" not in line
        assert "\n" not in line

    def test_literal_escape_text_is_not_double_escaped(self) -> None:
        reporter = MagicMock()
        result = Result(status=SafetyStatus.SAFE, summary=r"\x1b[31m")
        _write_result_line(
            terminalreporter=cast("TerminalReporter", reporter),
            result=result,
        )
        line = reporter.write_line.call_args[0][0]
        assert r"\x1b[31m" in line
        assert r"\\x1b" not in line


class TestTerminalSummary:
    """pytest_terminal_summary renders harm-category grouped output."""

    def _make_session_with_results(self) -> RampartSession:
        """Build a RampartSession with two results in different categories."""
        session = RampartSession()
        collector = ResultCollector()
        collector.record(
            result=Result(
                status=SafetyStatus.SAFE,
                summary="safe-one",
                harm_category="data_exfiltration",
            ),
        )
        collector.record(
            result=Result(
                status=SafetyStatus.UNSAFE,
                summary="unsafe-one",
                harm_category="jailbreak",
            ),
        )
        node = MagicMock()
        node.nodeid = "test_file.py::test_summary"
        session.absorb(node=node, collector=collector)
        return session

    def test_noop_when_no_session(self) -> None:
        reporter = MagicMock()
        config = MagicMock()
        config.stash = _StashStub()
        pytest_terminal_summary(
            terminalreporter=cast("TerminalReporter", reporter),
            exitstatus=0,
            config=cast("pytest.Config", config),
        )
        reporter.write_sep.assert_not_called()

    def test_noop_when_no_results(self) -> None:
        reporter = MagicMock()
        config = MagicMock()
        config.stash = _StashStub()
        from rampart.pytest_plugin.plugin import _rampart_key

        config.stash[_rampart_key] = RampartSession()
        pytest_terminal_summary(
            terminalreporter=cast("TerminalReporter", reporter),
            exitstatus=0,
            config=cast("pytest.Config", config),
        )
        reporter.write_sep.assert_not_called()

    def test_writes_incomplete_warning_even_without_results(self) -> None:
        reporter = MagicMock()
        config = MagicMock()
        config.stash = _StashStub()
        from rampart.pytest_plugin.plugin import _rampart_key

        session = RampartSession()
        reason = "worker gw0 crashed \x1b[31mred\nnext\x9b"
        session.mark_incomplete(reason=reason)
        config.stash[_rampart_key] = session
        pytest_terminal_summary(terminalreporter=reporter, exitstatus=0, config=config)

        sep_titles = [str(c) for c in reporter.write_sep.call_args_list]
        assert any("INCOMPLETE RUN" in t for t in sep_titles)
        reason_args = [
            c.args[0]
            for c in reporter.write_line.call_args_list
            if c.args and "gw0 crashed" in c.args[0]
        ]
        assert reason_args
        assert all("\x1b" not in arg for arg in reason_args)
        assert all("\n" not in arg for arg in reason_args)
        assert any(r"\x1b[31mred\x0anext\x9b" in arg for arg in reason_args)
        assert session.build_report().metadata["incomplete_reasons"] == [reason]

    def test_writes_summary_header(self) -> None:
        reporter = MagicMock()
        config = MagicMock()
        config.stash = _StashStub()
        from rampart.pytest_plugin.plugin import _rampart_key

        config.stash[_rampart_key] = self._make_session_with_results()
        pytest_terminal_summary(
            terminalreporter=cast("TerminalReporter", reporter),
            exitstatus=0,
            config=cast("pytest.Config", config),
        )
        reporter.write_sep.assert_called_once_with("=", "RAMPART Safety Summary")

    def test_writes_population_stats(self) -> None:
        reporter = MagicMock()
        config = MagicMock()
        config.stash = _StashStub()
        from rampart.pytest_plugin.plugin import _rampart_key

        config.stash[_rampart_key] = self._make_session_with_results()
        pytest_terminal_summary(
            terminalreporter=cast("TerminalReporter", reporter),
            exitstatus=0,
            config=cast("pytest.Config", config),
        )
        # Check that the Population line was written
        population_calls = [
            c for c in reporter.write_line.call_args_list if "Population:" in str(c)
        ]
        assert len(population_calls) == 1

    def test_controls_escaped_in_custom_harm_category(self) -> None:
        reporter = MagicMock()
        config = MagicMock()
        config.stash = _StashStub()
        from rampart.pytest_plugin.plugin import _rampart_key

        session = RampartSession()
        collector = ResultCollector()
        collector.record(
            result=Result(
                status=SafetyStatus.SAFE,
                summary="ok",
                harm_category="custom\x1b\nrisk\x9b",
            ),
        )
        node = MagicMock()
        node.nodeid = "test_file.py::test_category"
        session.absorb(node=node, collector=collector)
        config.stash[_rampart_key] = session

        pytest_terminal_summary(terminalreporter=reporter, exitstatus=0, config=config)

        heading = next(
            call.args[0]
            for call in reporter.write_line.call_args_list
            if "CUSTOM" in call.args[0]
        )
        assert r"CUSTOM\x1b\x0aRISK\x9b" in heading
        assert "\x1b" not in heading
        assert heading.count("\n") == 1


class TestRampartSessionSinks:
    """RampartSession accepts and exposes sinks."""

    def test_default_no_sinks(self) -> None:
        session = RampartSession()
        assert session.sinks == []

    def test_accepts_sinks(self) -> None:
        mock_sink = MagicMock()
        session = RampartSession(sinks=[mock_sink])
        assert len(session.sinks) == 1

    def test_sinks_returns_copy(self) -> None:
        mock_sink = MagicMock()
        session = RampartSession(sinks=[mock_sink])
        sinks = session.sinks
        sinks.clear()
        assert len(session.sinks) == 1


class TestRampartSessionAddSinks:
    """RampartSession.add_sinks merges fixture-provided sinks."""

    def test_add_sinks_appends(self) -> None:
        config_sink = MagicMock(spec=["emit_async"])
        config_sink.emit_async = MagicMock()
        session = RampartSession(sinks=[config_sink])

        fixture_sink = MagicMock(spec=["emit_async"])
        fixture_sink.emit_async = MagicMock()
        session.add_sinks(sinks=[fixture_sink])

        assert len(session.sinks) == 2

    def test_add_sinks_empty_list_noop(self) -> None:
        session = RampartSession()
        session.add_sinks(sinks=[])
        assert len(session.sinks) == 0

    def test_add_sinks_rejects_non_conforming(self) -> None:
        session = RampartSession()

        class NotASink:
            pass

        with pytest.raises(TypeError, match="Expected ReportSink"):
            session.add_sinks(sinks=[NotASink()])  # ty: ignore[invalid-argument-type]

    def test_add_sinks_preserves_existing(self) -> None:
        """Config-loaded sinks are not lost when fixture sinks are added."""
        sink_a = MagicMock(spec=["emit_async"])
        sink_a.emit_async = MagicMock()
        sink_b = MagicMock(spec=["emit_async"])
        sink_b.emit_async = MagicMock()

        session = RampartSession(sinks=[sink_a])
        session.add_sinks(sinks=[sink_b])

        assert session.sinks[0] is sink_a
        assert session.sinks[1] is sink_b


class TestRampartSessionDuration:
    """RampartSession tracks and reports duration."""

    def test_default_duration_zero(self) -> None:
        session = RampartSession()
        collector = ResultCollector()
        collector.record(
            result=Result(status=SafetyStatus.SAFE, summary="ok"),
        )
        node = MagicMock()
        node.nodeid = "test.py::test_dur"
        session.absorb(node=node, collector=collector)
        report = session.build_report()
        assert report.duration_seconds == pytest.approx(0.0)

    def test_set_duration_reflected_in_report(self) -> None:
        session = RampartSession()
        collector = ResultCollector()
        collector.record(
            result=Result(status=SafetyStatus.SAFE, summary="ok"),
        )
        node = MagicMock()
        node.nodeid = "test.py::test_dur"
        session.absorb(node=node, collector=collector)
        session.set_duration(duration_seconds=42.5)
        report = session.build_report()
        assert report.duration_seconds == pytest.approx(42.5)


class TestTrialGroupRendering:
    """Trial group aggregate lines are written to terminal."""

    def test_writes_trial_group_line(self) -> None:
        session = RampartSession()
        items: list[Any] = [MagicMock() for _ in range(10)]
        for idx, item in enumerate(items):
            item.nodeid = f"test_file.py::test_stat[trial-{idx}]"
            collector = ResultCollector()
            status = SafetyStatus.UNSAFE if idx < 2 else SafetyStatus.SAFE
            collector.record(
                result=Result(
                    status=status,
                    summary=f"t-{idx}",
                ),
            )
            session.absorb(node=item, collector=collector)

        session.record_trial_group(
            base_nodeid="test_file.py::test_stat",
            clone_nodeids=[item.nodeid for item in items],
            threshold=0.3,
        )

        reporter = MagicMock()
        _write_trial_group_lines(
            terminalreporter=cast("TerminalReporter", reporter),
            rampart_session=session,
        )

        reporter.write_line.assert_called_once()
        line = reporter.write_line.call_args[0][0]
        assert "8/10 safe" in line
        assert "80% pass rate" in line
        assert "FAILED" in line  # UNSAFE present → always fails

    def test_writes_passing_trial_group_line(self) -> None:
        session = RampartSession()
        items: list[Any] = [MagicMock() for _ in range(3)]
        for idx, item in enumerate(items):
            item.nodeid = f"test_file.py::test_pass[trial-{idx}]"
            collector = ResultCollector()
            collector.record(
                result=Result(
                    status=SafetyStatus.SAFE,
                    summary=f"t-{idx}",
                ),
            )
            session.absorb(node=item, collector=collector)

        session.record_trial_group(
            base_nodeid="test_file.py::test_pass",
            clone_nodeids=[item.nodeid for item in items],
            threshold=0.5,
        )

        reporter = MagicMock()
        _write_trial_group_lines(
            terminalreporter=cast("TerminalReporter", reporter),
            rampart_session=session,
        )

        reporter.write_line.assert_called_once()
        line = reporter.write_line.call_args[0][0]
        assert "3/3 safe" in line
        assert "100% pass rate" in line
        assert "PASSED" in line

    def test_no_trial_groups_writes_nothing(self) -> None:
        session = RampartSession()
        reporter = MagicMock()
        _write_trial_group_lines(
            terminalreporter=cast("TerminalReporter", reporter),
            rampart_session=session,
        )
        reporter.write_line.assert_not_called()

    def test_controls_escaped_in_trial_base_name(self) -> None:
        session = RampartSession()
        clone_nodeid = "test_file.py::test\x1b\ntrial\x9b[trial-0]"
        collector = ResultCollector()
        collector.record(result=Result(status=SafetyStatus.SAFE, summary="ok"))
        node = MagicMock()
        node.nodeid = clone_nodeid
        session.absorb(node=node, collector=collector)
        session.record_trial_group(
            base_nodeid="test_file.py::test\x1b\ntrial\x9b",
            clone_nodeids=[clone_nodeid],
            threshold=1.0,
        )
        reporter = MagicMock()

        _write_trial_group_lines(
            terminalreporter=cast("TerminalReporter", reporter),
            rampart_session=session,
        )

        line = reporter.write_line.call_args[0][0]
        assert r"test\x1b\x0atrial\x9b" in line
        assert "\x1b" not in line
        assert "\n" not in line


class TestEvaluateGates:
    """Gate evaluation logs when threshold is exceeded."""

    def test_logs_when_rate_exceeds_threshold(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session = RampartSession()
        items: list[Any] = [MagicMock() for _ in range(4)]
        for idx, item in enumerate(items):
            item.nodeid = f"test.py::test_gate[trial-{idx}]"
            collector = ResultCollector()
            status = SafetyStatus.UNSAFE if idx < 2 else SafetyStatus.SAFE
            collector.record(
                result=Result(
                    status=status,
                    summary=f"t-{idx}",
                ),
            )
            session.absorb(node=item, collector=collector)

        session.record_trial_group(
            base_nodeid="test.py::test\x1b\ngate\x9b",
            clone_nodeids=[item.nodeid for item in items],
            threshold=0.1,
        )

        with caplog.at_level(logging.INFO):
            _evaluate_gates(rampart_session=session)

        message = caplog.records[-1].getMessage()
        assert r"test.py::test\x1b\x0agate\x9b" in message
        assert "\x1b" not in message
        assert "\n" not in message


class TestTrialGateExitStatus:
    def _failed_group(self) -> RampartSession:
        rampart_session = RampartSession()
        rampart_session.record_trial_group(
            base_nodeid="test.py::test_gate",
            clone_nodeids=["test.py::test_gate[trial-0]"],
            threshold=1.0,
        )
        return rampart_session

    def test_failed_trial_forces_tests_failed(self) -> None:
        session = MagicMock()
        session.exitstatus = pytest.ExitCode.OK

        _enforce_trial_gate_exit_status(
            session=cast("pytest.Session", session),
            rampart_session=self._failed_group(),
        )

        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED

    @pytest.mark.parametrize(
        "exitstatus",
        [
            pytest.ExitCode.TESTS_FAILED,
            pytest.ExitCode.INTERRUPTED,
            pytest.ExitCode.INTERNAL_ERROR,
            pytest.ExitCode.USAGE_ERROR,
            pytest.ExitCode.NO_TESTS_COLLECTED,
            pytest.ExitCode.MAX_WARNINGS_ERROR,
        ],
    )
    def test_failed_trial_preserves_existing_nonzero_status(
        self,
        exitstatus: pytest.ExitCode,
    ) -> None:
        session = MagicMock()
        session.exitstatus = exitstatus

        _enforce_trial_gate_exit_status(
            session=cast("pytest.Session", session),
            rampart_session=self._failed_group(),
        )

        assert session.exitstatus == exitstatus

    def test_no_trial_groups_preserves_ok_status(self) -> None:
        session = MagicMock()
        session.exitstatus = pytest.ExitCode.OK

        _enforce_trial_gate_exit_status(
            session=cast("pytest.Session", session),
            rampart_session=RampartSession(),
        )

        assert session.exitstatus == pytest.ExitCode.OK


class TestTrialMarkerDeprecation:
    def test_warns_with_visible_pytest_category(self) -> None:
        rampart_session = RampartSession()
        rampart_session.register_trial_spec(
            clone_nodeid="test.py::test_x[trial-0]",
            base_nodeid="test.py::test_x",
            threshold=1.0,
        )

        with pytest.warns(
            pytest.PytestDeprecationWarning,
            match="forthcoming execution-domain trial API",
        ):
            _warn_trial_marker_deprecated(rampart_session=rampart_session)

    def test_noop_without_trial_specs(self, recwarn: pytest.WarningsRecorder) -> None:
        _warn_trial_marker_deprecated(rampart_session=RampartSession())
        assert not recwarn


class TestEmitSinks:
    """Sink emission calls emit_async and handles errors."""

    def test_noop_when_no_sinks(self) -> None:
        session = RampartSession()
        _emit_sinks(rampart_session=session)

    def test_sink_error_log_escapes_traceback_and_preserves_report(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A failing sink does not raise."""
        raw_error = "Kusto\x1b\t\n\r\x7f\x9b down"
        raw_summary = "summary\x1b\t\n\r\x7f\x9b"
        captured: list[str] = []

        class FailingSink:
            async def emit_async(self, *, report: TestRunReport) -> None:
                captured.append(report.results[0].summary)
                raise RuntimeError(raw_error)

        FailingSink.__name__ = "Sink\x1b\n\x9b"
        session = RampartSession(sinks=[FailingSink()])
        collector = ResultCollector()
        collector.record(
            result=Result(status=SafetyStatus.SAFE, summary=raw_summary),
        )
        node = MagicMock()
        node.nodeid = "test.py::test_sink"
        session.absorb(node=node, collector=collector)

        with caplog.at_level(logging.WARNING):
            _emit_sinks(rampart_session=session)

        formatted = _format_record(caplog.records[-1])
        assert r"Sink\x1b\x0a\x9b.emit_async failed" in formatted
        assert r"RuntimeError: Kusto\x1b\x09\x0a\x0d\x7f\x9b down" in formatted
        assert "Traceback (most recent call last):" in formatted
        assert "\x1b" not in formatted
        assert "\t" not in formatted
        assert "\n" not in formatted
        assert "\r" not in formatted
        assert "\x7f" not in formatted
        assert "\x9b" not in formatted
        assert captured == [raw_summary]

    def test_custom_sink_receives_raw_evidence(self) -> None:
        captured: list[str] = []

        class CapturingSink:
            async def emit_async(self, *, report: TestRunReport) -> None:
                captured.append(report.results[0].summary)

        raw_summary = "raw\x1b[31m\t\n\r\x7f\x9b雪"
        session = RampartSession(sinks=[CapturingSink()])
        collector = ResultCollector()
        collector.record(
            result=Result(status=SafetyStatus.UNSAFE, summary=raw_summary),
        )
        node = MagicMock()
        node.nodeid = "test.py::test_sink"
        session.absorb(node=node, collector=collector)

        _emit_sinks(rampart_session=session)

        assert captured == [raw_summary]


class TestSessionFinishIntegration:
    """pytest_sessionfinish aggregates trials, evaluates gates, and emits sinks."""

    def test_sets_duration(self) -> None:
        import time

        from rampart.pytest_plugin.plugin import (
            _rampart_key,
            _session_start_key,
        )

        session_mock = MagicMock()
        config_stash = _StashStub()
        rs = RampartSession()
        config_stash[_rampart_key] = rs
        config_stash[_session_start_key] = time.monotonic() - 5.0
        session_mock.config.stash = config_stash
        session_mock.items = []

        pytest_sessionfinish(session=cast("pytest.Session", session_mock), exitstatus=0)

        report = rs.build_report()
        assert report.duration_seconds >= 4.0


class TestSinkHookResolution:
    """The pytest_rampart_sinks hook is resolved and validated."""

    def test_has_sink_hook_impl_true_when_impls_present(self) -> None:
        config = MagicMock()
        hook = config.pluginmanager.hook.pytest_rampart_sinks
        hook.get_hookimpls.return_value = [MagicMock()]
        assert _has_sink_hook_impl(config=config) is True

    def test_has_sink_hook_impl_false_when_no_impls(self) -> None:
        config = MagicMock()
        hook = config.pluginmanager.hook.pytest_rampart_sinks
        hook.get_hookimpls.return_value = []
        assert _has_sink_hook_impl(config=config) is False

    def test_resolve_hook_sinks_flattens_implementations(self) -> None:
        sink_a = MagicMock(spec=ReportSink)
        sink_b = MagicMock(spec=ReportSink)
        config = MagicMock()
        config.pluginmanager.hook.pytest_rampart_sinks.return_value = [
            [sink_a],
            [sink_b],
        ]
        result = _resolve_hook_sinks(config=config)
        assert result == [sink_a, sink_b]

    def test_resolve_hook_sinks_drops_non_report_sinks(self) -> None:
        sink_a = MagicMock(spec=ReportSink)
        config = MagicMock()
        config.pluginmanager.hook.pytest_rampart_sinks.return_value = [
            [sink_a, "not-a-sink"],
        ]
        result = _resolve_hook_sinks(config=config)
        assert result == [sink_a]

    def test_resolve_hook_sinks_rejects_list_subclass_without_iteration(
        self,
    ) -> None:
        config = MagicMock()
        config.pluginmanager.hook.pytest_rampart_sinks.return_value = [
            _HostileList(),
        ]
        result = _resolve_hook_sinks(config=config)
        assert result == []

    def test_resolve_hook_sinks_does_not_repr_invalid_sink(self) -> None:
        config = MagicMock()
        config.pluginmanager.hook.pytest_rampart_sinks.return_value = [
            [_HostileRepr()],
        ]
        result = _resolve_hook_sinks(config=config)
        assert result == []

    def test_resolve_hook_sinks_skips_non_list_results(self) -> None:
        sink_a = MagicMock(spec=ReportSink)
        config = MagicMock()
        config.pluginmanager.hook.pytest_rampart_sinks.return_value = [
            "bad-impl-return",
            [sink_a],
        ]
        result = _resolve_hook_sinks(config=config)
        assert result == [sink_a]


class TestIncompleteExitStatus:
    """Incomplete runs are forced to a non-zero exit status."""

    def test_incomplete_run_forces_tests_failed(self) -> None:
        session = MagicMock()
        session.exitstatus = pytest.ExitCode.OK
        rampart_session = RampartSession()
        rampart_session.mark_incomplete(reason="worker gw1 crashed")
        _enforce_incomplete_exit_status(
            session=cast("pytest.Session", session),
            rampart_session=rampart_session,
        )
        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED

    def test_complete_run_preserves_ok_status(self) -> None:
        session = MagicMock()
        session.exitstatus = pytest.ExitCode.OK
        rampart_session = RampartSession()
        _enforce_incomplete_exit_status(
            session=cast("pytest.Session", session),
            rampart_session=rampart_session,
        )
        assert session.exitstatus == pytest.ExitCode.OK

    def test_incomplete_run_does_not_mask_existing_failure(self) -> None:
        session = MagicMock()
        session.exitstatus = pytest.ExitCode.INTERRUPTED
        rampart_session = RampartSession()
        rampart_session.mark_incomplete(reason="worker gw1 crashed")
        _enforce_incomplete_exit_status(
            session=cast("pytest.Session", session),
            rampart_session=rampart_session,
        )
        assert session.exitstatus == pytest.ExitCode.INTERRUPTED


def _make_result(*, summary: str = "result") -> Result:
    """Build a minimal Result for makereport tests."""
    return Result(status=SafetyStatus.SAFE, summary=summary)


def _make_reporting_item(*, worker: bool = True) -> Any:
    """Build a mock pytest.Item backed by a real Stash.

    Defaults to a worker-like config so the call-phase snapshot fires;
    pass worker=False for a single-process or controller config.
    """
    item = MagicMock()
    item.stash = pytest.Stash()
    if worker:
        item.config = SimpleNamespace(workerinput={"workerid": "gw0"})
    else:
        item.config = SimpleNamespace()
    return item


def _drive_makereport(*, item: Any, when: str, report: Any = None) -> Any:
    """Drive the makereport wrapper generator and return its result."""
    call = MagicMock()
    call.when = when
    sent = report if report is not None else MagicMock()
    gen = pytest_runtest_makereport(
        item=cast("pytest.Item", item),
        call=cast("pytest.CallInfo[None]", call),
    )
    next(gen)
    try:
        gen.send(sent)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("makereport wrapper did not return")


class TestPytestRuntestMakereport:
    """The makereport hook snapshots collector results at the call phase."""

    def test_snapshots_results_at_call_phase(self) -> None:
        item = _make_reporting_item()
        collector = ResultCollector()
        collector.record(result=_make_result(summary="captured"))
        token = activate_collector(collector)
        try:
            _drive_makereport(item=item, when="call")
        finally:
            deactivate_collector(token)

        snapshot = item.stash[_call_results_key]
        assert len(snapshot) == 1
        assert snapshot[0].summary == "captured"

    def test_snapshots_empty_list_when_no_results(self) -> None:
        item = _make_reporting_item()
        collector = ResultCollector()
        token = activate_collector(collector)
        try:
            _drive_makereport(item=item, when="call")
        finally:
            deactivate_collector(token)

        assert item.stash[_call_results_key] == []

    def test_no_snapshot_at_setup_phase(self) -> None:
        item = _make_reporting_item()
        collector = ResultCollector()
        collector.record(result=_make_result())
        token = activate_collector(collector)
        try:
            _drive_makereport(item=item, when="setup")
        finally:
            deactivate_collector(token)

        assert _call_results_key not in item.stash

    def test_no_snapshot_when_no_collector_active(self) -> None:
        item = _make_reporting_item()
        token = _active_collector.set(None)
        try:
            _drive_makereport(item=item, when="call")
        finally:
            _active_collector.reset(token)

        assert _call_results_key not in item.stash

    def test_no_snapshot_when_not_xdist_worker(self) -> None:
        item = _make_reporting_item(worker=False)
        collector = ResultCollector()
        collector.record(result=_make_result())
        token = activate_collector(collector)
        try:
            _drive_makereport(item=item, when="call")
        finally:
            deactivate_collector(token)

        assert _call_results_key not in item.stash

    def test_returns_report_unchanged(self) -> None:
        item = _make_reporting_item()
        report = object()

        returned = _drive_makereport(item=item, when="call", report=report)

        assert returned is report
