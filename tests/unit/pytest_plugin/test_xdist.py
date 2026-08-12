# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for shared xdist support and temporary clone metadata transport."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from typing import Any, Never, cast
from unittest.mock import MagicMock

import pytest
from _pytest.config import PytestPluginManager

from rampart.core.result import HarmCategory, InjectionRecord, Result, SafetyStatus
from rampart.core.trial import (
    TRIAL_BATCH_COUNT_KEY,
    TRIAL_BATCH_ID_KEY,
    TRIAL_BATCH_INDEX_KEY,
    TRIAL_BATCH_SCHEMA,
    TRIAL_BATCH_SCHEMA_KEY,
    TRIAL_BATCH_THRESHOLD_KEY,
)
from rampart.core.types import (
    EvalOutcome,
    EvalResult,
    ObservabilityLevel,
    Payload,
    Request,
    Response,
    SideEffect,
    ToolCall,
    Turn,
)
from rampart.pytest_plugin._session import RampartSession, TrialSpec
from rampart.pytest_plugin._xdist import (
    DEFAULT_SIZE_LIMIT_BYTES,
    MAX_METADATA_DEPTH,
    SIZE_LIMIT_OPTION,
    TRIAL_SPECS_SCHEMA_VERSION,
    TRIAL_SPECS_WORKEROUTPUT_KEY,
    TrialSpecValidationError,
    WorkerOutputError,
    _deserialize_result,
    _registered_plugin_name,
    _serialize_result,
    _to_json_safe,
    deserialize_trial_specs,
    discover_sinks_from_conftest,
    finalize_trial_specs_worker,
    get_dist_mode,
    get_worker_count,
    is_xdist_controller,
    is_xdist_worker,
)
from rampart.reporting.sink import ReportSink


class _HostileRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr failed")

    def __str__(self) -> str:
        raise RuntimeError("str failed")


class _HostileString(str):  # ruff: ignore[subclass-builtin]
    __slots__ = ()

    def __len__(self) -> int:
        raise RuntimeError("len failed")

    def __getitem__(self, key: object) -> str:
        del key
        raise RuntimeError("getitem failed")

    def __repr__(self) -> str:
        raise RuntimeError("repr failed")


class _HostileList(list):  # ruff: ignore[subclass-builtin]
    def __iter__(self) -> Never:
        raise RuntimeError("iter failed")

    def __len__(self) -> int:
        raise RuntimeError("len failed")


class _HostilePlugin:
    rampart_sinks: object = object()

    def __repr__(self) -> str:
        raise RuntimeError("repr failed")


class _StatefulEqPlugin:
    __hash__ = object.__hash__

    def __init__(self, *, rampart_sinks: list[object]) -> None:
        self.rampart_sinks = rampart_sinks
        self.equality_calls = 0
        self.raise_on_equality = False

    def __eq__(self, other: object) -> bool:
        self.equality_calls += 1
        if self.raise_on_equality:
            raise AssertionError("plugin equality must not be invoked")
        return self is other


class _HostileError(Exception):
    def __str__(self) -> str:
        raise SystemExit("str invoked")


def _make_plugin_config(*, plugins: list[tuple[object, object]]) -> MagicMock:
    config = MagicMock()
    config.pluginmanager.list_name_plugin.return_value = plugins
    return config


def _make_result(
    *,
    status: SafetyStatus = SafetyStatus.SAFE,
    summary: str = "summary",
    harm_category: HarmCategory | str | None = None,
    metadata: dict[str, Any] | None = None,
    turns: list[Turn] | None = None,
    injections: list[InjectionRecord] | None = None,
    observability_level: ObservabilityLevel = ObservabilityLevel.RESPONSE_ONLY,
) -> Result:
    return Result(
        status=status,
        summary=summary,
        turns=turns or [],
        harm_category=harm_category,
        strategy="xpia",
        observability_level=observability_level,
        injections=injections or [],
        metadata=metadata or {},
    )


def _make_config(
    *,
    is_worker: bool = False,
    numprocesses: int | None = None,
    dist: str = "no",
    tx: list[str] | None = None,
    max_bytes: int | None = None,
) -> Any:
    config = MagicMock()
    if is_worker:
        config.workerinput = {"workerid": "worker-a"}
    else:
        del config.workerinput
    config.option = MagicMock()
    config.option.numprocesses = numprocesses
    config.option.dist = dist
    config.option.tx = tx

    def _getoption(name: str, default: object = None) -> object:
        return max_bytes if name == SIZE_LIMIT_OPTION else default

    def _getini(name: str) -> None:
        del name

    config.getoption = _getoption
    config.getini = _getini
    return config


def _round_trip(result: Result, *, nodeid: str = "test.py::test_item") -> Result:
    payload = _serialize_result(result=result, nodeid=nodeid)
    return _deserialize_result(data=json.loads(json.dumps(payload)))


def _valid_trial_payload(*, specs: object = None) -> dict[str, object]:
    return {
        "schema": TRIAL_SPECS_SCHEMA_VERSION,
        "trial_specs": [] if specs is None else specs,
    }


def _format_record(record: logging.LogRecord) -> str:
    return logging.Formatter("%(levelname)s:%(message)s").format(record)


class TestDetection:
    def test_worker_detection_uses_workerinput(self) -> None:
        assert is_xdist_worker(config=_make_config(is_worker=True)) is True
        assert is_xdist_worker(config=_make_config()) is False

    def test_controller_detection_uses_active_endpoints(self) -> None:
        assert (
            is_xdist_controller(
                config=_make_config(numprocesses=2, dist="load"),
            )
            is True
        )
        assert (
            is_xdist_controller(
                config=_make_config(dist="load", tx=["popen"]),
            )
            is True
        )

    def test_controller_detection_excludes_workers_and_inactive_dist(self) -> None:
        assert (
            is_xdist_controller(
                config=_make_config(is_worker=True, numprocesses=2, dist="load"),
            )
            is False
        )
        assert is_xdist_controller(config=_make_config(dist="load")) is False
        assert (
            is_xdist_controller(
                config=_make_config(numprocesses=2, dist="no"),
            )
            is False
        )

    def test_dist_mode_and_worker_count(self) -> None:
        config = _make_config(numprocesses=3, dist="loadgroup")
        assert get_dist_mode(config=config) == "loadgroup"
        assert get_worker_count(config=config) == 3


class TestToJsonSafe:
    def test_primitives_and_nested_collections_round_trip(self) -> None:
        value = {"a": [1, "two", True, None], "b": (3, 4)}
        assert _to_json_safe(value=value) == {
            "a": [1, "two", True, None],
            "b": [3, 4],
        }

    def test_nonfinite_float_becomes_none(self) -> None:
        assert _to_json_safe(value=math.nan) is None
        assert _to_json_safe(value=math.inf) is None

    def test_custom_value_uses_repr(self) -> None:
        class Value:
            def __repr__(self) -> str:
                return "<value>"

        assert _to_json_safe(value=Value()) == "<value>"

    def test_depth_limit_remains_json_safe(self) -> None:
        nested: dict[str, Any] = {"value": "leaf"}
        for _ in range(MAX_METADATA_DEPTH + 2):
            nested = {"value": nested}
        json.dumps(_to_json_safe(value=nested))


class TestResultCodec:
    def test_status_observability_and_harm_round_trip(self) -> None:
        for status in SafetyStatus:
            for level in ObservabilityLevel:
                result = _round_trip(
                    _make_result(
                        status=status,
                        harm_category=HarmCategory.JAILBREAK,
                        observability_level=level,
                    ),
                )
                assert result.status is status
                assert result.observability_level is level
                assert result.harm_category is HarmCategory.JAILBREAK

    def test_nested_evidence_round_trip(self) -> None:
        timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        attachment = Payload(content="payload\n\x1b", metadata={"order": [2, 1]})
        tool_call = ToolCall(
            name="send_email",
            arguments={"to": "a@example.test"},
            result="sent",
            timestamp=timestamp,
        )
        side_effect = SideEffect(kind="http", details={"status": 201})
        evaluation = EvalResult(
            outcome=EvalOutcome.DETECTED,
            confidence=0.9,
            evidence=["line\none", "\x1braw"],
            rationale="because",
        )
        turn = Turn(
            request=Request(prompt="prompt", attachments=[attachment]),
            response=Response(
                text="response",
                tool_calls=[tool_call],
                side_effects=[side_effect],
                metadata={"nested": {"flag": True}},
            ),
            eval_result=evaluation,
            timestamp=timestamp,
            driver_reasoning="reasoning",
        )
        original = _make_result(
            summary="summary\n\x1b",
            turns=[turn],
            injections=[InjectionRecord(payload_id="p1", surface_name="OneDrive")],
            metadata={"user": {"values": [1, 1, 2]}},
        )

        recovered = _round_trip(original)

        assert recovered.summary == original.summary
        assert recovered.turns[0].request.attachments[0].metadata == {"order": [2, 1]}
        assert recovered.turns[0].response.tool_calls[0].arguments == {
            "to": "a@example.test"
        }
        assert recovered.turns[0].response.side_effects[0].details == {"status": 201}
        assert recovered.turns[0].eval_result is not None
        assert recovered.turns[0].eval_result.evidence == ["line\none", "\x1braw"]
        assert recovered.injections[0].surface_name == "OneDrive"
        assert recovered.metadata == original.metadata

    def test_trial_batch_metadata_round_trip(self) -> None:
        metadata = {
            TRIAL_BATCH_SCHEMA_KEY: TRIAL_BATCH_SCHEMA,
            TRIAL_BATCH_ID_KEY: "123e4567-e89b-42d3-a456-426614174000",
            TRIAL_BATCH_INDEX_KEY: 0,
            TRIAL_BATCH_COUNT_KEY: 1,
            TRIAL_BATCH_THRESHOLD_KEY: 1.0,
            "_rampart_result_index": 7,
        }
        assert _round_trip(_make_result(metadata=metadata)).metadata == metadata

    def test_codec_payload_is_pure_json(self) -> None:
        payload = _serialize_result(
            result=_make_result(metadata={"tuple": (1, 2)}),
            nodeid="test.py::test_json",
        )
        assert json.loads(json.dumps(payload))["metadata"]["tuple"] == [1, 2]

    def test_invalid_enum_is_rejected(self) -> None:
        payload = _serialize_result(result=_make_result(), nodeid="n")
        payload["status"] = "unknown"
        with pytest.raises(WorkerOutputError, match="SafetyStatus"):
            _deserialize_result(data=payload)


class TestTrialSpecs:
    def test_finalize_and_deserialize_round_trip(self) -> None:
        config = _make_config(is_worker=True)
        workeroutput: dict[str, object] = {}
        config.workeroutput = workeroutput
        session = RampartSession()
        session.register_trial_spec(
            clone_nodeid="t.py::test_a[trial-0]",
            base_nodeid="t.py::test_a",
            threshold=0.75,
        )

        finalize_trial_specs_worker(config=config, session=session)
        payload = workeroutput[TRIAL_SPECS_WORKEROUTPUT_KEY]

        assert deserialize_trial_specs(data=json.loads(json.dumps(payload))) == {
            "t.py::test_a[trial-0]": TrialSpec(
                base_nodeid="t.py::test_a",
                threshold=0.75,
            )
        }

    def test_empty_payload_is_valid(self) -> None:
        assert deserialize_trial_specs(data=_valid_trial_payload()) == {}

    @pytest.mark.parametrize(
        "payload",
        [
            "not-a-dict",
            {},
            {"schema": "rampart.xdist.v1", "trial_specs": []},
            {
                "schema": TRIAL_SPECS_SCHEMA_VERSION,
                "trial_specs": [],
                "results_by_nodeid": {},
            },
            {"schema": TRIAL_SPECS_SCHEMA_VERSION, "trial_specs": "not-a-list"},
        ],
    )
    def test_rejects_unknown_or_bulk_payloads(self, payload: object) -> None:
        with pytest.raises(WorkerOutputError):
            deserialize_trial_specs(data=payload)

    @pytest.mark.parametrize(
        "specs",
        [
            ["not-a-dict"],
            [{"clone_nodeid": "", "base_nodeid": "b", "threshold": 0.5}],
            [{"clone_nodeid": "a", "base_nodeid": 3, "threshold": 0.5}],
            [
                {
                    "clone_nodeid": "a",
                    "base_nodeid": "b",
                    "threshold": 0.5,
                    "extra": True,
                }
            ],
            [
                {"clone_nodeid": "a", "base_nodeid": "b", "threshold": 0.5},
                {"clone_nodeid": "a", "base_nodeid": "b", "threshold": 0.5},
            ],
        ],
    )
    def test_rejects_malformed_or_duplicate_entries(self, specs: object) -> None:
        with pytest.raises(TrialSpecValidationError):
            deserialize_trial_specs(data=_valid_trial_payload(specs=specs))

    @pytest.mark.parametrize(
        "threshold",
        [None, "0.5", True, 0.0, -0.1, 1.1, math.inf, math.nan, 10**400],
    )
    def test_rejects_invalid_threshold(self, threshold: object) -> None:
        spec = {
            "clone_nodeid": "a",
            "base_nodeid": "b",
            "threshold": threshold,
        }
        with pytest.raises(TrialSpecValidationError, match="finite number"):
            deserialize_trial_specs(data=_valid_trial_payload(specs=[spec]))

    def test_explicit_worker_error_is_bounded_and_rejected(self) -> None:
        payload = {
            **_valid_trial_payload(),
            "rampart_transport_error": "x" * 1000,
        }
        with pytest.raises(TrialSpecValidationError) as exc_info:
            deserialize_trial_specs(data=payload)
        assert len(str(exc_info.value)) < 300

    def test_merge_is_idempotent_and_conflicts_do_not_overwrite(self) -> None:
        session = RampartSession()
        original = TrialSpec(base_nodeid="b1", threshold=0.5)
        replacement = TrialSpec(base_nodeid="b2", threshold=0.9)

        assert session.merge_trial_specs(trial_specs={"k": original}) == ()
        assert session.merge_trial_specs(trial_specs={"k": original}) == ()
        assert session.merge_trial_specs(trial_specs={"k": replacement}) == ("k",)
        assert session.trial_specs == {"k": original}

    def test_finalize_writes_error_without_results(self) -> None:
        config = _make_config(is_worker=True)
        workeroutput: dict[str, object] = {}
        config.workeroutput = workeroutput
        session = RampartSession()
        session.append_transported_results(
            nodeid="n",
            source_worker="worker-a",
            sequence=1,
            results=((0, _make_result(summary="must-not-be-serialized")),),
        )
        session.register_trial_spec(
            clone_nodeid="a",
            base_nodeid="b",
            threshold=0.0,
        )

        with pytest.raises(TrialSpecValidationError):
            finalize_trial_specs_worker(config=config, session=session)

        payload = workeroutput[TRIAL_SPECS_WORKEROUTPUT_KEY]
        assert isinstance(payload, dict)
        typed_payload = cast("dict[str, object]", payload)
        assert "results_by_nodeid" not in typed_payload
        assert typed_payload["trial_specs"] == []
        assert "rampart_transport_error" in typed_payload

    def test_finalize_is_noop_on_controller(self) -> None:
        config = _make_config(numprocesses=2, dist="load")
        config.workeroutput = {}
        finalize_trial_specs_worker(config=config, session=RampartSession())
        assert config.workeroutput == {}


class TestSinkDiscovery:
    def test_finds_callable_rampart_sinks(self) -> None:
        sink = MagicMock(spec=ReportSink)
        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=lambda: [sink],
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])
        result = discover_sinks_from_conftest(config=config)
        assert sink in result

    def test_finds_list_rampart_sinks(self) -> None:
        sink = MagicMock(spec=ReportSink)
        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=[sink],
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])
        result = discover_sinks_from_conftest(config=config)
        assert sink in result

    def test_real_manager_does_not_invoke_plugin_equality(self) -> None:
        sink = MagicMock(spec=ReportSink)
        plugin = _StatefulEqPlugin(rampart_sinks=[sink])
        plugin_manager = PytestPluginManager()
        plugin_manager.register(plugin, name="hostile-eq")
        plugin.equality_calls = 0
        plugin.raise_on_equality = True
        config = MagicMock()
        config.pluginmanager = plugin_manager

        result = discover_sinks_from_conftest(config=config)

        assert result == [sink]
        assert plugin.equality_calls == 0

    def test_unmatched_plugin_name_uses_type_fallback_by_identity(self) -> None:
        plugin = _StatefulEqPlugin(rampart_sinks=[])
        plugin.raise_on_equality = True

        result = _registered_plugin_name(
            registered_plugins=[("different", object())],
            plugin=plugin,
        )

        assert result == "_StatefulEqPlugin"
        assert plugin.equality_calls == 0

    def test_non_exact_registered_name_uses_type_fallback(self) -> None:
        plugin = _StatefulEqPlugin(rampart_sinks=[])
        plugin.raise_on_equality = True

        result = _registered_plugin_name(
            registered_plugins=[(_HostileString("hostile"), plugin)],
            plugin=plugin,
        )

        assert result == "_StatefulEqPlugin"
        assert plugin.equality_calls == 0

    def test_invalid_candidate_does_not_repr_plugin(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugin = _HostilePlugin()
        config = _make_plugin_config(plugins=[("hostile", plugin)])
        with caplog.at_level(logging.WARNING):
            result = discover_sinks_from_conftest(config=config)
        assert result == []
        assert "controller-side discovery" in caplog.records[-1].getMessage()

    def test_list_subclass_is_not_iterated(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugin = MagicMock()
        plugin.rampart_sinks = _HostileList()
        config = _make_plugin_config(plugins=[("mod", plugin)])
        with caplog.at_level(logging.WARNING):
            result = discover_sinks_from_conftest(config=config)
        assert result == []

    def test_factory_exception_uses_safe_traceback(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def failing_sinks() -> list[ReportSink]:
            raise _HostileError

        plugin = MagicMock()
        plugin.rampart_sinks = failing_sinks
        config = _make_plugin_config(plugins=[("mod", plugin)])
        with caplog.at_level(logging.WARNING):
            assert discover_sinks_from_conftest(config=config) == []
        formatted = _format_record(caplog.records[-1])
        assert "Traceback (most recent call last):" in formatted
        assert "_HostileError: <exception str() failed>" in formatted

    def test_returns_empty_when_no_rampart_sinks(self) -> None:
        plugin = MagicMock(spec=["__name__"], __name__="mod")
        config = _make_plugin_config(plugins=[("mod", plugin)])
        result = discover_sinks_from_conftest(config=config)
        assert result == []

    def test_warns_on_callable_with_required_args(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def needs_arg(other: object) -> list[ReportSink]:
            return []

        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=needs_arg,
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])
        with caplog.at_level(logging.WARNING):
            result = discover_sinks_from_conftest(config=config)
        assert result == []
        assert any("requires arguments" in r.getMessage() for r in caplog.records)

    def test_resolves_parameterless_fixture_form(self) -> None:
        sink = MagicMock(spec=ReportSink)

        @pytest.fixture
        def rampart_sinks() -> list[ReportSink]:
            return [sink]

        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=rampart_sinks,
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])
        result = discover_sinks_from_conftest(config=config)
        assert sink in result

    def test_warns_and_skips_fixture_with_dependencies(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        @pytest.fixture
        def rampart_sinks(tmp_path: object) -> list[ReportSink]:
            return []

        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=rampart_sinks,
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])
        with caplog.at_level(logging.WARNING):
            result = discover_sinks_from_conftest(config=config)
        assert result == []
        assert any("requires arguments" in r.getMessage() for r in caplog.records)
        assert any("pytest_rampart_sinks" in r.getMessage() for r in caplog.records)

    def test_callable_exception_log_escapes_complete_traceback(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        raw_error = "legacy\x1b\t\n\r\x7f\x9b"
        error = RuntimeError(raw_error)

        def rampart_sinks() -> list[ReportSink]:
            raise error

        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=rampart_sinks,
            __name__="plugin\x1b\n\x9b",
        )
        config = _make_plugin_config(plugins=[("plugin\x1b\n\x9b", plugin)])

        with caplog.at_level(logging.WARNING):
            result = discover_sinks_from_conftest(config=config)

        assert result == []
        formatted = _format_record(caplog.records[-1])
        assert r"plugin\x1b\x0a\x9b" in formatted
        assert r"RuntimeError: legacy\x1b\x09\x0a\x0d\x7f\x9b" in formatted
        assert "Traceback (most recent call last):" in formatted
        assert "\x1b" not in formatted
        assert "\t" not in formatted
        assert "\n" not in formatted
        assert "\r" not in formatted
        assert "\x7f" not in formatted
        assert "\x9b" not in formatted
        assert str(error) == raw_error

    def test_non_report_sink_repr_is_not_invoked(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=[_HostileRepr()],
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])

        with caplog.at_level(logging.WARNING):
            result = discover_sinks_from_conftest(config=config)

        assert result == []
        formatted = _format_record(caplog.records[-1])
        assert "<_HostileRepr>" in formatted


class TestSinkDeprecationWarning:
    """Deprecation-warning contract for controller-side ``rampart_sinks`` discovery.

    The ``@pytest.fixture`` form warns when resolved; the module-level list form
    is not a fixture and must stay silent. These fast, in-process checks replace
    the equivalent ``pytester`` subprocess test in ``test_xdist_aggregation.py``.
    """

    def test_fixture_form_emits_deprecation_warning(self) -> None:
        sink = MagicMock(spec=ReportSink)

        @pytest.fixture
        def rampart_sinks() -> list[ReportSink]:
            return [sink]

        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=rampart_sinks,
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])
        with pytest.warns(
            DeprecationWarning,
            match="rampart_sinks fixture is deprecated",
        ):
            result = discover_sinks_from_conftest(config=config)
        assert sink in result

    def test_list_form_does_not_emit_deprecation_warning(
        self,
        recwarn: pytest.WarningsRecorder,
    ) -> None:
        sink = MagicMock(spec=ReportSink)
        plugin = MagicMock(
            spec=["rampart_sinks", "__name__"],
            rampart_sinks=[sink],
            __name__="mod",
        )
        config = _make_plugin_config(plugins=[("mod", plugin)])
        result = discover_sinks_from_conftest(config=config)
        assert sink in result
        assert not any(
            issubclass(w.category, DeprecationWarning)
            and "rampart_sinks fixture is deprecated" in str(w.message)
            for w in recwarn
        )


class TestConstants:
    def test_envelope_limit_defaults_to_64mb(self) -> None:
        assert DEFAULT_SIZE_LIMIT_BYTES == 64 * 1024 * 1024

    def test_trial_workeroutput_is_narrowly_namespaced(self) -> None:
        assert TRIAL_SPECS_SCHEMA_VERSION == "rampart.xdist.trial-specs.v1"
        assert TRIAL_SPECS_WORKEROUTPUT_KEY == "rampart_xdist_trial_specs_v1"
