# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the xdist v2 shadow transport."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import rampart.pytest_plugin._xdist_shadow as shadow_module
from rampart.core.result import Result, SafetyStatus
from rampart.core.types import (
    ObservabilityLevel,
    Payload,
    PayloadFormat,
    Request,
    Response,
    Turn,
)
from rampart.pytest_plugin._collection import ResultCollector
from rampart.pytest_plugin._session import RampartSession
from rampart.pytest_plugin._xdist import DEFAULT_SIZE_LIMIT_BYTES, SIZE_LIMIT_OPTION
from rampart.pytest_plugin._xdist_shadow import (
    REPORT_ATTRIBUTE,
    SHADOW_MANIFEST_KEY,
    SHADOW_SCHEMA_VERSION,
    EnvelopeBuild,
    ShadowSizeError,
    ShadowTransportError,
    XdistShadowRuntime,
    _freeze_json,
    _result_key,
    build_envelope,
    parse_envelope,
    validate_manifest,
)


class _Config:
    def __init__(
        self,
        *,
        worker: bool = False,
        controller: bool = False,
        max_bytes: int = DEFAULT_SIZE_LIMIT_BYTES,
    ) -> None:
        self.option = SimpleNamespace(
            dist="load" if worker or controller else "no",
            numprocesses=2 if controller else 0,
            tx=None,
        )
        self.workeroutput: dict[str, object] = {}
        self._max_bytes = max_bytes
        if worker:
            self.workerinput = {"workerid": "worker-a"}

    def getoption(self, name: str, default: object = None) -> object:
        return self._max_bytes if name == SIZE_LIMIT_OPTION else default

    def getini(self, name: str) -> None:
        del name


def _make_result(
    *,
    summary: str = "summary",
    metadata: dict[str, Any] | None = None,
) -> Result:
    return Result(
        status=SafetyStatus.SAFE,
        summary=summary,
        observability_level=ObservabilityLevel.RESPONSE_ONLY,
        metadata=metadata or {},
    )


def _make_report(
    *,
    encoded: object,
    nodeid: str = "test_shadow.py::test_item",
    worker_id: object = "worker-a",
) -> pytest.TestReport:
    report = pytest.TestReport(
        nodeid=nodeid,
        location=("test_shadow.py", 1, "test_item"),
        keywords={},
        outcome="passed",
        longrepr=None,
        when="teardown",
    )
    report.__dict__["worker_id"] = worker_id
    setattr(report, REPORT_ATTRIBUTE, encoded)
    return report


def _make_manifest(
    *,
    envelopes_sent: int = 1,
    last_sequence: int = 1,
    results_sent: int = 1,
    results_dropped: int = 0,
) -> dict[str, int]:
    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "envelopes_sent": envelopes_sent,
        "last_sequence": last_sequence,
        "results_sent": results_sent,
        "results_dropped": results_dropped,
    }


def _make_node(
    *,
    manifest: object | None = None,
    worker_id: object = "worker-a",
) -> SimpleNamespace:
    workeroutput = {}
    if manifest is not None:
        workeroutput[SHADOW_MANIFEST_KEY] = manifest
    return SimpleNamespace(
        gateway=SimpleNamespace(id=worker_id),
        workeroutput=workeroutput,
    )


def _controller_runtime(
    *,
    max_bytes: int = DEFAULT_SIZE_LIMIT_BYTES,
) -> tuple[_Config, RampartSession, XdistShadowRuntime]:
    config = _Config(controller=True, max_bytes=max_bytes)
    session = RampartSession()
    return config, session, _make_runtime(config=config, session=session)


def _make_runtime(
    *,
    config: _Config,
    session: RampartSession,
) -> XdistShadowRuntime:
    typed_config: Any = config
    return XdistShadowRuntime(config=typed_config, session=session)


def _finish_runtime(*, runtime: XdistShadowRuntime, config: _Config) -> None:
    pytest_session: Any = SimpleNamespace(config=config)
    runtime.pytest_sessionfinish(pytest_session, 0)


def _run_worker_makereport(
    *,
    runtime: XdistShadowRuntime,
    item: MagicMock,
) -> pytest.TestReport:
    report = _make_report(encoded="")
    delattr(report, REPORT_ATTRIBUTE)
    wrapper = runtime.pytest_runtest_makereport(item, MagicMock())
    next(wrapper)
    with pytest.raises(StopIteration) as stopped:
        wrapper.send(report)
    assert stopped.value.value is report
    return report


class TestEnvelopeBuild:
    def test_zero_results_emit_no_envelope(self) -> None:
        built = build_envelope(
            results=(),
            nodeid="test.py::test_zero",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )

        assert built == EnvelopeBuild(
            encoded=None,
            results_sent=0,
            results_dropped=0,
        )

    def test_multiple_results_preserve_order_and_indexes(self) -> None:
        built = build_envelope(
            results=(
                _make_result(summary="first"),
                _make_result(summary="second"),
            ),
            nodeid="test.py::test_many",
            sequence=4,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )

        assert built.encoded is not None
        envelope = parse_envelope(
            encoded=built.encoded,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
            nodeid="test.py::test_many",
        )
        assert [item.index for item in envelope.results] == [0, 1]
        assert [item.result.summary for item in envelope.results] == [
            "first",
            "second",
        ]
        assert envelope.dropped == ()

    def test_later_same_node_attempt_preserves_cumulative_indexes(self) -> None:
        session = RampartSession()
        node = MagicMock()
        node.nodeid = "test.py::test_rerun"
        node.get_closest_marker.return_value = None
        first = ResultCollector()
        first.record(result=_make_result(summary="first-attempt"))
        second = ResultCollector()
        second.record(result=_make_result(summary="later-0"))
        second.record(result=_make_result(summary="later-1"))
        session.absorb(node=node, collector=first)
        absorbed = session.absorb(node=node, collector=second)

        built = build_envelope(
            results=absorbed,
            nodeid=node.nodeid,
            sequence=2,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )

        assert built.encoded is not None
        raw = json.loads(built.encoded)
        assert [record["index"] for record in raw["results"]] == [0, 1]
        assert [
            record["result"]["metadata"]["_rampart_result_index"]
            for record in raw["results"]
        ] == [1, 2]
        envelope = parse_envelope(
            encoded=built.encoded,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
            nodeid=node.nodeid,
        )
        assert [record.index for record in envelope.results] == [0, 1]
        assert [
            record.result.metadata["_rampart_result_index"]
            for record in envelope.results
        ] == [1, 2]

    def test_individually_oversized_result_keeps_siblings(self) -> None:
        limit = 1000
        built = build_envelope(
            results=(
                _make_result(summary="first"),
                _make_result(summary="x" * 5000),
                _make_result(summary="third"),
            ),
            nodeid="test.py::test_size",
            sequence=1,
            limit=limit,
        )

        assert built.encoded is not None
        assert len(built.encoded.encode("utf-8")) <= limit
        envelope = parse_envelope(
            encoded=built.encoded,
            limit=limit,
            nodeid="test.py::test_size",
        )
        assert [item.index for item in envelope.results] == [0, 2]
        assert [item.result.summary for item in envelope.results] == [
            "first",
            "third",
        ]
        assert [item.index for item in envelope.dropped] == [1]

    def test_drop_size_includes_structural_result_wrapper(self) -> None:
        result = _make_result(summary="x" * 5000)
        roomy = build_envelope(
            results=(result,),
            nodeid="test.py::test_wrapper_size",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert roomy.encoded is not None
        roomy_raw = json.loads(roomy.encoded)
        record = roomy_raw["results"][0]

        constrained = build_envelope(
            results=(result,),
            nodeid="test.py::test_wrapper_size",
            sequence=1,
            limit=1000,
        )

        assert constrained.encoded is not None
        constrained_raw = json.loads(constrained.encoded)
        [drop] = constrained_raw["dropped"]
        assert drop["serialized_bytes"] == shadow_module._encoded_size(record)
        assert drop["serialized_bytes"] > shadow_module._encoded_size(record["result"])

    def test_combined_cap_retains_deterministic_prefix(self) -> None:
        results = tuple(_make_result(summary=str(index) * 120) for index in range(4))
        outcomes = [
            (
                limit,
                build_envelope(
                    results=results,
                    nodeid="test.py::test_prefix",
                    sequence=1,
                    limit=limit,
                ),
            )
            for limit in range(300, 1800)
        ]
        limit, built = next(
            (limit, outcome)
            for limit, outcome in outcomes
            if outcome.encoded is not None and 0 < outcome.results_sent < len(results)
        )

        assert built.encoded is not None
        assert len(built.encoded.encode("utf-8")) <= limit
        envelope = parse_envelope(
            encoded=built.encoded,
            limit=limit,
            nodeid="test.py::test_prefix",
        )
        retained = [item.index for item in envelope.results]
        assert retained == list(range(len(retained)))
        assert [item.index for item in envelope.dropped] == list(
            range(len(retained), len(results))
        )

    def test_all_drop_envelope_that_cannot_fit_is_not_attached(self) -> None:
        built = build_envelope(
            results=(_make_result(summary="large"),),
            nodeid="test.py::test_tiny",
            sequence=1,
            limit=1,
        )

        assert built.encoded is None
        assert built.results_sent == 0
        assert built.results_dropped == 1

    def test_utf8_bytes_not_character_count_control_limit(self) -> None:
        result = _make_result(summary="雪" * 50)
        roomy = build_envelope(
            results=(result,),
            nodeid="test.py::test_unicode",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert roomy.encoded is not None
        exact_size = len(roomy.encoded.encode("utf-8"))

        constrained = build_envelope(
            results=(result,),
            nodeid="test.py::test_unicode",
            sequence=1,
            limit=exact_size - 1,
        )

        assert constrained.results_sent == 0
        if constrained.encoded is not None:
            assert len(constrained.encoded.encode("utf-8")) <= exact_size - 1

    def test_candidate_sizing_is_linear_in_encoded_volume(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original = shadow_module._encode_json
        encoded_volume = 0

        def _measured(value: object) -> str:
            nonlocal encoded_volume
            encoded = original(value)
            encoded_volume += len(encoded.encode("utf-8"))
            return encoded

        monkeypatch.setattr(shadow_module, "_encode_json", _measured)
        built = build_envelope(
            results=tuple(_make_result(summary=str(index)) for index in range(200)),
            nodeid="test.py::test_linear",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )

        assert built.encoded is not None
        assert encoded_volume < len(built.encoded.encode("utf-8")) * 5


class TestEnvelopeValidation:
    def test_unknown_schema_is_rejected(self) -> None:
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_schema",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        raw["schema_version"] = 99

        with pytest.raises(ShadowTransportError, match="unsupported"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_schema",
            )

    @pytest.mark.parametrize(
        "encoded",
        [
            "{",
            '{"schema_version":2,"schema_version":2}',
            '{"schema_version":NaN}',
        ],
    )
    def test_malformed_json_is_rejected(self, encoded: str) -> None:
        with pytest.raises(ShadowTransportError):
            parse_envelope(
                encoded=encoded,
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_json",
            )

    def test_attribute_size_is_guarded_before_parsing(self) -> None:
        with pytest.raises(ShadowSizeError, match="10-byte cap"):
            parse_envelope(
                encoded="x" * 11,
                limit=10,
                nodeid="test.py::test_size",
            )

    def test_invalid_utf8_text_is_rejected_without_escaping(self) -> None:
        with pytest.raises(ShadowTransportError, match="UTF-8"):
            parse_envelope(
                encoded="\ud800",
                limit=10,
                nodeid="test.py::test_utf8",
            )

    def test_bool_sequence_is_rejected(self) -> None:
        raw = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "sequence": True,
            "produced": 1,
            "results": [],
            "dropped": [{"index": 0, "reason": "size_limit", "serialized_bytes": 10}],
        }

        with pytest.raises(ShadowTransportError, match="sequence"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_bool",
            )

    def test_produced_count_mismatch_is_rejected(self) -> None:
        raw = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "sequence": 1,
            "produced": 2,
            "results": [],
            "dropped": [{"index": 0, "reason": "size_limit", "serialized_bytes": 10}],
        }

        with pytest.raises(ShadowTransportError, match="produced"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_count",
            )

    def test_duplicate_result_indexes_are_rejected(self) -> None:
        built = build_envelope(
            results=(_make_result(summary="a"), _make_result(summary="b")),
            nodeid="test.py::test_indexes",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        raw["results"][1]["index"] = 0

        with pytest.raises(ShadowTransportError, match="partition"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_indexes",
            )

    @pytest.mark.parametrize(
        "mutation",
        [
            "missing-index",
            "extra-key",
            "bool-index",
            "negative-index",
            "nondict-result",
        ],
    )
    def test_result_record_shape_is_strict(self, mutation: str) -> None:
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_record",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        record = raw["results"][0]
        if mutation == "missing-index":
            del record["index"]
        elif mutation == "extra-key":
            record["unexpected"] = "value"
        elif mutation == "bool-index":
            record["index"] = True
        elif mutation == "negative-index":
            record["index"] = -1
        else:
            record["result"] = []

        with pytest.raises(ShadowTransportError):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_record",
            )

    def test_result_record_indexes_must_be_increasing(self) -> None:
        built = build_envelope(
            results=(_make_result(summary="a"), _make_result(summary="b")),
            nodeid="test.py::test_result_order",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        raw["results"][0]["index"] = 1
        raw["results"][1]["index"] = 0

        with pytest.raises(ShadowTransportError, match="increasing"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_result_order",
            )

    def test_drop_indexes_must_be_increasing(self) -> None:
        raw = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "sequence": 1,
            "produced": 2,
            "results": [],
            "dropped": [
                {"index": 1, "reason": "size_limit", "serialized_bytes": 10},
                {"index": 0, "reason": "size_limit", "serialized_bytes": 10},
            ],
        }

        with pytest.raises(ShadowTransportError, match="drop indexes"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_drop_order",
            )

    @pytest.mark.parametrize("drop_index", [0, 2], ids=["duplicate", "gap"])
    def test_result_and_drop_indexes_must_partition(self, drop_index: int) -> None:
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_partition",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        raw["produced"] = 2
        raw["dropped"] = [
            {
                "index": drop_index,
                "reason": "size_limit",
                "serialized_bytes": 10,
            }
        ]

        with pytest.raises(ShadowTransportError, match="partition"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_partition",
            )

    @pytest.mark.parametrize("key", ["_pytest_nodeid", "_rampart_source_worker"])
    def test_authoritative_route_metadata_is_rejected(self, key: str) -> None:
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_route",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        raw["results"][0]["result"]["metadata"][key] = "spoofed"

        with pytest.raises(ShadowTransportError, match="route metadata"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_route",
            )

    def test_result_coercion_is_rejected(self) -> None:
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_coercion",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        raw["results"][0]["result"]["summary"] = 7

        with pytest.raises(ShadowTransportError, match="summary"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_coercion",
            )

    def test_unknown_result_field_is_rejected(self) -> None:
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_extra",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        raw["results"][0]["result"]["unexpected"] = "value"

        with pytest.raises(ShadowTransportError, match="exactly"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_extra",
            )

    def test_valid_non_text_payload_survives_v1_normalization(self) -> None:
        result = Result(
            status=SafetyStatus.SAFE,
            summary="html",
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
        built = build_envelope(
            results=(result,),
            nodeid="test.py::test_html",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None

        envelope = parse_envelope(
            encoded=built.encoded,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
            nodeid="test.py::test_html",
        )

        attachment = envelope.results[0].result.turns[0].request.attachments[0]
        assert attachment.format is PayloadFormat.TEXT
        assert attachment.metadata["_rampart_worker_format"] == "html"

    def test_valid_nonfinite_duration_normalization_is_accepted(self) -> None:
        result = _make_result()
        result.duration_seconds = float("nan")
        built = build_envelope(
            results=(result,),
            nodeid="test.py::test_duration",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None

        envelope = parse_envelope(
            encoded=built.encoded,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
            nodeid="test.py::test_duration",
        )

        assert envelope.results[0].result.duration_seconds == pytest.approx(0.0)

    def test_deep_metadata_is_rejected_without_recursion_error(self) -> None:
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_depth",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        nested: object = "leaf"
        for _ in range(600):
            nested = [nested]
        raw["results"][0]["result"]["metadata"]["deep"] = nested

        with pytest.raises(ShadowTransportError, match="depth"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_depth",
            )

    def test_v1_maximum_metadata_depth_is_accepted(self) -> None:
        nested: object = "leaf"
        for _ in range(7):
            nested = [nested]
        built = build_envelope(
            results=(_make_result(metadata={"nested": nested}),),
            nodeid="test.py::test_max_depth",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None

        envelope = parse_envelope(
            encoded=built.encoded,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
            nodeid="test.py::test_max_depth",
        )

        assert len(envelope.results) == 1

    def test_unknown_drop_reason_is_rejected(self) -> None:
        raw = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "sequence": 1,
            "produced": 1,
            "results": [],
            "dropped": [{"index": 0, "reason": "other", "serialized_bytes": 10}],
        }

        with pytest.raises(ShadowTransportError, match="unsupported"):
            parse_envelope(
                encoded=json.dumps(raw),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_reason",
            )


class TestSchedulingIndexValidation:
    def test_absent_scheduling_index_round_trips_without_synthesis(self) -> None:
        nodeid = "test.py::test_absent_scheduling_index"
        result = _make_result()
        built = build_envelope(
            results=(result,),
            nodeid=nodeid,
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        raw = json.loads(built.encoded)
        assert "_rampart_result_index" not in raw["results"][0]["result"]["metadata"]

        _, session, runtime = _controller_runtime()
        runtime.pytest_runtest_logreport(
            _make_report(encoded=built.encoded, nodeid=nodeid)
        )

        delivery = next(iter(runtime._controller.deliveries.values()))
        [validated] = delivery.envelope.results
        assert "_rampart_result_index" not in validated.result.metadata
        assert _result_key(nodeid=nodeid, result=result) == _result_key(
            nodeid=nodeid,
            result=validated.result,
        )
        assert session.is_incomplete is False

    @pytest.mark.parametrize(
        "value",
        [True, -1, "0", 0.0, None],
        ids=["bool", "negative", "string", "float", "none"],
    )
    def test_present_scheduling_index_is_strict(self, value: object) -> None:
        built = build_envelope(
            results=(_make_result(metadata={"_rampart_result_index": value}),),
            nodeid="test.py::test_scheduling_index",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None

        with pytest.raises(ShadowTransportError, match="scheduling index"):
            parse_envelope(
                encoded=built.encoded,
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid="test.py::test_scheduling_index",
            )


class TestManifestValidation:
    def test_zero_manifest_is_valid(self) -> None:
        manifest = validate_manifest(
            _make_manifest(
                envelopes_sent=0,
                last_sequence=0,
                results_sent=0,
            )
        )

        assert manifest.last_sequence == 0

    @pytest.mark.parametrize(
        "mutation",
        [
            {"schema_version": 3},
            {"envelopes_sent": True},
            {"last_sequence": -1},
            {"envelopes_sent": 2, "last_sequence": 1},
        ],
    )
    def test_malformed_manifest_is_rejected(
        self,
        mutation: dict[str, object],
    ) -> None:
        raw: dict[str, object] = dict(_make_manifest())
        raw.update(mutation)

        with pytest.raises(ShadowTransportError):
            validate_manifest(raw)

    def test_missing_manifest_field_is_rejected(self) -> None:
        raw = _make_manifest()
        del raw["results_sent"]

        with pytest.raises(ShadowTransportError, match="exactly"):
            validate_manifest(raw)


class TestWorkerRuntime:
    def test_teardown_attaches_envelope_and_advances_sequence(self) -> None:
        config = _Config(worker=True)
        runtime = _make_runtime(config=config, session=RampartSession())
        item = MagicMock()
        item.nodeid = "test.py::test_attempt"
        item.stash = pytest.Stash()

        runtime.remember_results(item=item, results=(_make_result(summary="one"),))
        first = _run_worker_makereport(runtime=runtime, item=item)
        runtime.remember_results(item=item, results=(_make_result(summary="two"),))
        second = _run_worker_makereport(runtime=runtime, item=item)

        assert (
            parse_envelope(
                encoded=getattr(first, REPORT_ATTRIBUTE),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid=item.nodeid,
            ).sequence
            == 1
        )
        assert (
            parse_envelope(
                encoded=getattr(second, REPORT_ATTRIBUTE),
                limit=DEFAULT_SIZE_LIMIT_BYTES,
                nodeid=item.nodeid,
            ).sequence
            == 2
        )
        assert len(item.stash) == 0

    def test_zero_result_attempt_does_not_advance_sequence(self) -> None:
        config = _Config(worker=True)
        runtime = _make_runtime(config=config, session=RampartSession())
        item = MagicMock()
        item.nodeid = "test.py::test_zero"
        item.stash = pytest.Stash()

        runtime.remember_results(item=item, results=())
        report = _run_worker_makereport(runtime=runtime, item=item)
        _finish_runtime(runtime=runtime, config=config)

        assert not hasattr(report, REPORT_ATTRIBUTE)
        assert config.workeroutput[SHADOW_MANIFEST_KEY] == _make_manifest(
            envelopes_sent=0,
            last_sequence=0,
            results_sent=0,
        )

    def test_tiny_cap_records_unsent_drop_without_sequence(self) -> None:
        config = _Config(worker=True, max_bytes=1)
        runtime = _make_runtime(config=config, session=RampartSession())
        item = MagicMock()
        item.nodeid = "test.py::test_tiny"
        item.stash = pytest.Stash()

        runtime.remember_results(item=item, results=(_make_result(),))
        report = _run_worker_makereport(runtime=runtime, item=item)
        _finish_runtime(runtime=runtime, config=config)

        assert not hasattr(report, REPORT_ATTRIBUTE)
        assert config.workeroutput[SHADOW_MANIFEST_KEY] == _make_manifest(
            envelopes_sent=0,
            last_sequence=0,
            results_sent=0,
            results_dropped=1,
        )


class TestControllerRuntime:
    def test_duplicate_delivery_is_idempotent(self) -> None:
        _, session, runtime = _controller_runtime()
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_duplicate",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        report = _make_report(
            encoded=built.encoded,
            nodeid="test.py::test_duplicate",
        )

        runtime.pytest_runtest_logreport(report)
        runtime.pytest_runtest_logreport(report)

        assert len(runtime._controller.deliveries) == 1
        assert session.is_incomplete is False

    def test_semantically_equal_duplicate_json_is_idempotent(self) -> None:
        _, session, runtime = _controller_runtime()
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_duplicate_json",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        reformatted = json.dumps(json.loads(built.encoded), indent=2)

        runtime.pytest_runtest_logreport(
            _make_report(
                encoded=built.encoded,
                nodeid="test.py::test_duplicate_json",
            )
        )
        runtime.pytest_runtest_logreport(
            _make_report(
                encoded=reformatted,
                nodeid="test.py::test_duplicate_json",
            )
        )

        assert len(runtime._controller.deliveries) == 1
        assert session.is_incomplete is False

    def test_conflicting_duplicate_fails_closed(self) -> None:
        _, session, runtime = _controller_runtime()
        first = build_envelope(
            results=(_make_result(summary="first"),),
            nodeid="test.py::test_conflict",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        second = build_envelope(
            results=(_make_result(summary="second"),),
            nodeid="test.py::test_conflict",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert first.encoded is not None
        assert second.encoded is not None

        runtime.pytest_runtest_logreport(
            _make_report(
                encoded=first.encoded,
                nodeid="test.py::test_conflict",
            )
        )
        runtime.pytest_runtest_logreport(
            _make_report(
                encoded=second.encoded,
                nodeid="test.py::test_conflict",
            )
        )

        assert len(runtime._controller.deliveries) == 1
        assert session.is_incomplete is True
        assert any("conflicting" in reason for reason in session.incomplete_reasons)

    def test_conflicting_structural_slot_assignment_fails_closed(self) -> None:
        _, session, runtime = _controller_runtime()
        first = build_envelope(
            results=(
                _make_result(summary="slot-a"),
                _make_result(summary="slot-b"),
            ),
            nodeid="test.py::test_slot_conflict",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        second = build_envelope(
            results=(
                _make_result(summary="slot-b"),
                _make_result(summary="slot-a"),
            ),
            nodeid="test.py::test_slot_conflict",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert first.encoded is not None
        assert second.encoded is not None

        runtime.pytest_runtest_logreport(
            _make_report(
                encoded=first.encoded,
                nodeid="test.py::test_slot_conflict",
            )
        )
        runtime.pytest_runtest_logreport(
            _make_report(
                encoded=second.encoded,
                nodeid="test.py::test_slot_conflict",
            )
        )

        assert len(runtime._controller.deliveries) == 1
        assert session.is_incomplete is True
        assert any("conflicting" in reason for reason in session.incomplete_reasons)

    @pytest.mark.parametrize("worker_id", [None, "", 7])
    def test_invalid_worker_identity_marks_incomplete(self, worker_id: object) -> None:
        _, session, runtime = _controller_runtime()

        runtime.pytest_runtest_logreport(
            _make_report(encoded="{}", worker_id=worker_id)
        )

        assert session.is_incomplete is True

    @pytest.mark.parametrize("nodeid", ["", 7])
    def test_invalid_node_identity_marks_incomplete(self, nodeid: object) -> None:
        _, session, runtime = _controller_runtime()
        report = _make_report(encoded="{}")
        report.__dict__["nodeid"] = nodeid

        runtime.pytest_runtest_logreport(report)

        assert session.is_incomplete is True

    def test_oversized_attribute_marks_incomplete(self) -> None:
        _, session, runtime = _controller_runtime(max_bytes=10)

        runtime.pytest_runtest_logreport(_make_report(encoded="x" * 11))

        assert session.is_incomplete is True
        assert runtime._controller.deliveries == {}

    def test_unexpected_validation_error_is_contained(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, session, runtime = _controller_runtime()

        def _fail(**kwargs: object) -> None:
            del kwargs
            raise RuntimeError("unexpected validator failure")

        monkeypatch.setattr(shadow_module, "parse_envelope", _fail)

        runtime.pytest_runtest_logreport(_make_report(encoded="{}"))

        assert session.is_incomplete is True
        assert any(
            code.startswith("envelope-boundary-error:")
            for code in runtime._controller.fault_codes
        )

    def test_out_of_order_sequences_reconcile_with_manifest(self) -> None:
        config, session, runtime = _controller_runtime()
        nodeids = ["test.py::test_first", "test.py::test_second"]
        for sequence, nodeid in [(2, nodeids[1]), (1, nodeids[0])]:
            built = build_envelope(
                results=(_make_result(summary=nodeid),),
                nodeid=nodeid,
                sequence=sequence,
                limit=DEFAULT_SIZE_LIMIT_BYTES,
            )
            assert built.encoded is not None
            runtime.pytest_runtest_logreport(
                _make_report(encoded=built.encoded, nodeid=nodeid)
            )
            session.merge_worker_results(
                results_by_nodeid={nodeid: [_make_result(summary=nodeid)]}
            )
        runtime.pytest_testnodedown(
            _make_node(
                manifest=_make_manifest(
                    envelopes_sent=2,
                    last_sequence=2,
                    results_sent=2,
                )
            ),
            None,
        )
        _finish_runtime(runtime=runtime, config=config)

        assert session.is_incomplete is False

    def test_sequence_gap_marks_incomplete(self) -> None:
        _, session, runtime = _controller_runtime()
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_gap",
            sequence=2,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        runtime.pytest_runtest_logreport(_make_report(encoded=built.encoded))

        runtime.pytest_testnodedown(
            _make_node(
                manifest=_make_manifest(
                    envelopes_sent=2,
                    last_sequence=2,
                )
            ),
            None,
        )

        assert session.is_incomplete is True
        assert any("sequence" in reason for reason in session.incomplete_reasons)

    def test_manifest_result_count_mismatch_marks_incomplete(self) -> None:
        _, session, runtime = _controller_runtime()
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_count",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        runtime.pytest_runtest_logreport(_make_report(encoded=built.encoded))

        runtime.pytest_testnodedown(
            _make_node(
                manifest=_make_manifest(results_sent=2),
            ),
            None,
        )

        assert session.is_incomplete is True
        assert any("Result count" in reason for reason in session.incomplete_reasons)

    def test_manifest_drop_count_mismatch_marks_incomplete(self) -> None:
        _, session, runtime = _controller_runtime()
        built = build_envelope(
            results=(_make_result(),),
            nodeid="test.py::test_drop_count",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        runtime.pytest_runtest_logreport(_make_report(encoded=built.encoded))

        runtime.pytest_testnodedown(
            _make_node(
                manifest=_make_manifest(results_dropped=1),
            ),
            None,
        )

        assert session.is_incomplete is True
        assert any("drop count" in reason for reason in session.incomplete_reasons)

    def test_clean_missing_manifest_marks_incomplete(self) -> None:
        _, session, runtime = _controller_runtime()

        runtime.pytest_testnodedown(_make_node(), None)

        assert session.is_incomplete is True
        assert any("missing" in reason for reason in session.incomplete_reasons)

    def test_malformed_manifest_marks_incomplete(self) -> None:
        _, session, runtime = _controller_runtime()

        runtime.pytest_testnodedown(
            _make_node(manifest={"schema_version": SHADOW_SCHEMA_VERSION}),
            None,
        )

        assert session.is_incomplete is True
        assert any("manifest" in reason for reason in session.incomplete_reasons)

    def test_worker_loss_preserves_delivered_shadow_results(self) -> None:
        _, session, runtime = _controller_runtime()
        built = build_envelope(
            results=(_make_result(summary="delivered"),),
            nodeid="test.py::test_delivered",
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        runtime.pytest_runtest_logreport(
            _make_report(
                encoded=built.encoded,
                nodeid="test.py::test_delivered",
            )
        )

        runtime.pytest_testnodedown(_make_node(), RuntimeError("worker lost"))

        assert len(runtime._controller.deliveries) == 1
        delivery = next(iter(runtime._controller.deliveries.values()))
        assert delivery.envelope.results[0].result.summary == "delivered"
        assert session.is_incomplete is True

    def test_drop_record_preserves_survivors_and_fails_closed(self) -> None:
        _, session, runtime = _controller_runtime(max_bytes=1000)
        built = build_envelope(
            results=(
                _make_result(summary="first"),
                _make_result(summary="x" * 5000),
                _make_result(summary="third"),
            ),
            nodeid="test.py::test_drop",
            sequence=1,
            limit=1000,
        )
        assert built.encoded is not None

        runtime.pytest_runtest_logreport(
            _make_report(encoded=built.encoded, nodeid="test.py::test_drop")
        )

        delivery = next(iter(runtime._controller.deliveries.values()))
        assert [item.result.summary for item in delivery.envelope.results] == [
            "first",
            "third",
        ]
        assert session.is_incomplete is True

    def test_reconciliation_preserves_cumulative_scheduling_index(self) -> None:
        config, session, runtime = _controller_runtime()
        nodeid = "test.py::test_cumulative_index"
        result = _make_result(
            summary="same",
            metadata={"_rampart_result_index": 7},
        )
        built = build_envelope(
            results=(result,),
            nodeid=nodeid,
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        runtime.pytest_runtest_logreport(
            _make_report(encoded=built.encoded, nodeid=nodeid)
        )
        runtime.pytest_testnodedown(
            _make_node(manifest=_make_manifest()),
            None,
        )
        session.merge_worker_results(
            results_by_nodeid={
                nodeid: [
                    _make_result(
                        summary="same",
                        metadata={"_rampart_result_index": 7},
                    )
                ]
            }
        )

        _finish_runtime(runtime=runtime, config=config)

        delivery = next(iter(runtime._controller.deliveries.values()))
        [validated] = delivery.envelope.results
        assert validated.index == 0
        assert validated.result.metadata["_rampart_result_index"] == 7
        assert session.is_incomplete is False

    def test_semantic_divergence_marks_incomplete(self) -> None:
        config, session, runtime = _controller_runtime()
        nodeid = "test.py::test_divergence"
        built = build_envelope(
            results=(_make_result(summary="v2"),),
            nodeid=nodeid,
            sequence=1,
            limit=DEFAULT_SIZE_LIMIT_BYTES,
        )
        assert built.encoded is not None
        runtime.pytest_runtest_logreport(
            _make_report(encoded=built.encoded, nodeid=nodeid)
        )
        runtime.pytest_testnodedown(
            _make_node(manifest=_make_manifest()),
            None,
        )
        session.merge_worker_results(
            results_by_nodeid={nodeid: [_make_result(summary="v1")]}
        )

        _finish_runtime(runtime=runtime, config=config)

        assert session.is_incomplete is True
        assert any("diverged" in reason for reason in session.incomplete_reasons)


class TestSemanticNormalization:
    def test_transport_metadata_is_excluded(self) -> None:
        left = _make_result(
            metadata={
                "_pytest_nodeid": "left",
                "_rampart_result_index": 1,
                "_rampart_source_worker": "worker-left",
                "evidence": "same",
            }
        )
        right = _make_result(
            metadata={
                "_pytest_nodeid": "right",
                "_rampart_result_index": 9,
                "_rampart_source_worker": "worker-right",
                "evidence": "same",
            }
        )

        assert _result_key(nodeid="test.py::test_key", result=left) == _result_key(
            nodeid="test.py::test_key",
            result=right,
        )

    def test_trial_batch_metadata_remains_semantically_significant(self) -> None:
        left = _make_result(
            metadata={
                "_rampart_result_index": 0,
                "_rampart_trial_batch_index": 0,
            }
        )
        right = _make_result(
            metadata={
                "_rampart_result_index": 9,
                "_rampart_trial_batch_index": 1,
            }
        )

        assert _result_key(
            nodeid="test.py::test_trial_key",
            result=left,
        ) != _result_key(nodeid="test.py::test_trial_key", result=right)

    def test_textual_evidence_difference_is_preserved(self) -> None:
        left = _make_result(summary="line\none")
        right = _make_result(summary="line\n two")

        assert _result_key(
            nodeid="test.py::test_text",
            result=left,
        ) != _result_key(nodeid="test.py::test_text", result=right)

    def test_list_order_and_multiplicity_are_preserved(self) -> None:
        assert _freeze_json(["a", "b"]) != _freeze_json(["b", "a"])
        assert _freeze_json(["a", "a"]) != _freeze_json(["a"])

    def test_bool_int_and_float_do_not_collide(self) -> None:
        assert (
            len(
                {
                    _freeze_json(value=True),
                    _freeze_json(value=1),
                    _freeze_json(value=1.0),
                }
            )
            == 3
        )
