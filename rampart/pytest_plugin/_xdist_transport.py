# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Authoritative incremental xdist Result transport.

The v2 channel in this module validates and incrementally appends per-item
Results before clean worker shutdown.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypedDict, cast

import pytest

from rampart.core.result import Result, SafetyStatus
from rampart.core.types import EvalOutcome, ObservabilityLevel, PayloadFormat
from rampart.pytest_plugin._diagnostics import (
    bounded_text,
    safe_type_name,
)
from rampart.pytest_plugin._xdist import (
    MAX_METADATA_DEPTH,
    TRIAL_SPECS_WORKEROUTPUT_KEY,
    WorkerOutputError,
    _deserialize_result,
    _serialize_result,
    _size_limit,
    deserialize_trial_specs,
    is_xdist_controller,
    is_xdist_worker,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from rampart.pytest_plugin._session import RampartSession

logger = logging.getLogger(__name__)

REPORT_ATTRIBUTE: str = "_rampart_xdist_v2_json"
MANIFEST_KEY: str = "rampart_xdist_v2_manifest"
SCHEMA_VERSION: int = 2
RUNTIME_PLUGIN_NAME: str = "_rampart_xdist_v2_transport"

_NODEID_METADATA_KEY: str = "_pytest_nodeid"
_RESULT_INDEX_METADATA_KEY: str = "_rampart_result_index"
_SOURCE_WORKER_METADATA_KEY: str = "_rampart_source_worker"
_EVAL_OUTCOME_VALUES = frozenset(item.value for item in EvalOutcome)
_OBSERVABILITY_VALUES = frozenset(item.value for item in ObservabilityLevel)
_PAYLOAD_FORMAT_VALUES = frozenset(item.value for item in PayloadFormat)
_SAFETY_STATUS_VALUES = frozenset(item.value for item in SafetyStatus)


class DropReason(StrEnum):
    """Reasons a Result can be omitted from an envelope."""

    SIZE_LIMIT = "size_limit"


class DropWire(TypedDict):
    """Wire representation of one omitted Result."""

    index: int
    reason: str
    serialized_bytes: int


class EvalResultWire(TypedDict):
    """Wire projection of one evaluator result."""

    outcome: str
    confidence: float | None
    evidence: list[str]
    rationale: str


class PayloadWire(TypedDict):
    """Wire projection of one request payload."""

    content: str
    id: str
    format: str
    artifact: str | None
    metadata: dict[str, Any]


class RequestWire(TypedDict):
    """Wire projection of one request."""

    prompt: str | None
    attachments: list[PayloadWire]


class ToolCallWire(TypedDict):
    """Wire projection of one tool call."""

    name: str
    arguments: dict[str, Any]
    result: str | None
    timestamp: str | None


class SideEffectWire(TypedDict):
    """Wire projection of one side effect."""

    kind: str
    details: dict[str, Any]


class ResponseWire(TypedDict):
    """Wire projection of one response."""

    text: str
    tool_calls: list[ToolCallWire]
    side_effects: list[SideEffectWire]
    metadata: dict[str, Any]


class TurnWire(TypedDict):
    """Wire projection of one conversation turn."""

    request: RequestWire
    response: ResponseWire
    eval_result: EvalResultWire | None
    turn_number: int
    timestamp: str | None
    driver_reasoning: str


class InjectionWire(TypedDict):
    """Wire projection of one injection record."""

    payload_id: str | None
    surface_name: str


class ResultWire(TypedDict):
    """Exact v1 Result projection carried inside a v2 envelope."""

    safe: bool
    status: str
    summary: str
    turns: list[TurnWire]
    duration_seconds: float | None
    harm_category: str | None
    strategy: str
    observability_level: str
    injections: list[InjectionWire]
    metadata: dict[str, Any]


class ResultRecordWire(TypedDict):
    """One successful Result with its envelope-local slot."""

    index: int
    result: ResultWire


class EnvelopeWire(TypedDict):
    """Per-item v2 Result envelope."""

    schema_version: int
    sequence: int
    produced: int
    results: list[ResultRecordWire]
    dropped: list[DropWire]


class ManifestWire(TypedDict):
    """O(1) clean-finish delivery manifest."""

    schema_version: int
    envelopes_sent: int
    last_sequence: int
    results_sent: int
    results_dropped: int


class TransportError(Exception):
    """Raised when v2 data violates the transport contract."""


class TransportSizeError(TransportError):
    """Raised before parsing an oversized report attribute."""


@dataclass(frozen=True, kw_only=True)
class _SerializedResult:
    index: int
    record: ResultRecordWire
    serialized_bytes: int
    drop: DropWire
    drop_bytes: int


@dataclass(frozen=True, kw_only=True)
class EnvelopeBuild:
    """Outcome of constructing one worker report attribute."""

    encoded: str | None
    results_sent: int
    results_dropped: int


@dataclass(frozen=True, kw_only=True)
class ValidatedResult:
    """One strictly validated Result and its envelope-local identity."""

    index: int
    result: Result


@dataclass(frozen=True, kw_only=True)
class ValidatedDrop:
    """One strictly validated drop record."""

    index: int
    serialized_bytes: int


@dataclass(frozen=True, kw_only=True)
class ValidatedEnvelope:
    """Validated semantic envelope used by controller state."""

    sequence: int
    produced: int
    results: tuple[ValidatedResult, ...]
    dropped: tuple[ValidatedDrop, ...]
    semantic_key: object


@dataclass(frozen=True, kw_only=True)
class ValidatedManifest:
    """Validated clean-finish worker counters."""

    envelopes_sent: int
    last_sequence: int
    results_sent: int
    results_dropped: int
    semantic_key: object


@dataclass(frozen=True, kw_only=True)
class AcceptedDelivery:
    """One unique controller-side envelope delivery."""

    worker_id: str
    nodeid: str
    envelope: ValidatedEnvelope


@dataclass(kw_only=True)
class _WorkerState:
    last_sequence: int = 0
    envelopes_sent: int = 0
    results_sent: int = 0
    results_dropped: int = 0

    def manifest(self) -> ManifestWire:
        """Return the worker's constant-size clean-finish manifest."""
        return {
            "schema_version": SCHEMA_VERSION,
            "envelopes_sent": self.envelopes_sent,
            "last_sequence": self.last_sequence,
            "results_sent": self.results_sent,
            "results_dropped": self.results_dropped,
        }


@dataclass(kw_only=True)
class _ControllerState:
    deliveries: dict[tuple[str, int], AcceptedDelivery] = field(default_factory=dict)
    manifests: dict[str, ValidatedManifest] = field(default_factory=dict)
    report_workers: set[str] = field(default_factory=set)
    node_down_workers: set[str] = field(default_factory=set)
    fault_codes: set[str] = field(default_factory=set)
    reconciled: bool = False


ITEM_RESULTS_KEY = pytest.StashKey[tuple[Result, ...]]()


def _encode_json(value: object) -> str:
    """Return one deterministic, non-lossy JSON wire encoding."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _encoded_size(value: object) -> int:
    """Return the UTF-8 byte length of the actual wire encoding."""
    return len(_encode_json(value).encode("utf-8"))


def _prepare_result(
    *,
    result: Result,
    nodeid: str,
    index: int,
) -> _SerializedResult:
    """Return one structural record without authoritative route metadata."""
    raw = _serialize_result(result=result, nodeid=nodeid)
    metadata = dict(cast("dict[str, Any]", raw["metadata"]))
    metadata.pop(_NODEID_METADATA_KEY, None)
    metadata.pop(_SOURCE_WORKER_METADATA_KEY, None)
    raw["metadata"] = metadata
    payload = cast("ResultWire", raw)
    record = ResultRecordWire(index=index, result=payload)
    record_encoded = _encode_json(record)
    serialized_bytes = len(record_encoded.encode("utf-8"))
    drop = _drop_wire(index=index, serialized_bytes=serialized_bytes)
    drop_encoded = _encode_json(drop)
    return _SerializedResult(
        index=index,
        record=record,
        serialized_bytes=serialized_bytes,
        drop=drop,
        drop_bytes=len(drop_encoded.encode("utf-8")),
    )


def _drop_wire(*, index: int, serialized_bytes: int) -> DropWire:
    """Return the bounded drop marker for one Result."""
    return {
        "index": index,
        "reason": DropReason.SIZE_LIMIT.value,
        "serialized_bytes": serialized_bytes,
    }


def _envelope_wire(
    *,
    sequence: int,
    records: Sequence[_SerializedResult],
    retained: frozenset[int],
) -> EnvelopeWire:
    """Return one complete envelope for a selected Result subset."""
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "produced": len(records),
        "results": [record.record for record in records if record.index in retained],
        "dropped": [record.drop for record in records if record.index not in retained],
    }


def build_envelope(
    *,
    results: tuple[Result, ...],
    nodeid: str,
    sequence: int,
    limit: int,
) -> EnvelopeBuild:
    """Return a deterministic fitting Result subset in one attribute."""
    if not results:
        return EnvelopeBuild(encoded=None, results_sent=0, results_dropped=0)
    records = tuple(
        _prepare_result(result=result, nodeid=nodeid, index=index)
        for index, result in enumerate(results)
    )
    retained: set[int] = set()
    all_drop = _envelope_wire(
        sequence=sequence,
        records=records,
        retained=frozenset(),
    )
    current_size = _encoded_size(all_drop)
    if current_size > limit:
        return EnvelopeBuild(
            encoded=None,
            results_sent=0,
            results_dropped=len(records),
        )

    retained_count = 0
    dropped_count = len(records)
    for record in records:
        if record.serialized_bytes > limit:
            continue
        candidate_size = (
            current_size
            + record.serialized_bytes
            + int(retained_count > 0)
            - record.drop_bytes
            - int(dropped_count > 1)
        )
        if candidate_size <= limit:
            retained.add(record.index)
            retained_count += 1
            dropped_count -= 1
            current_size = candidate_size

    envelope = _envelope_wire(
        sequence=sequence,
        records=records,
        retained=frozenset(retained),
    )
    encoded = _encode_json(envelope)
    if len(encoded.encode("utf-8")) > limit:
        return EnvelopeBuild(
            encoded=None,
            results_sent=0,
            results_dropped=len(records),
        )
    return EnvelopeBuild(
        encoded=encoded,
        results_sent=len(retained),
        results_dropped=len(records) - len(retained),
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Return a JSON object after rejecting duplicate keys.

    Raises:
        TransportError: If the object repeats a key.
    """
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            msg = "V2 JSON objects must not contain duplicate keys."
            raise TransportError(msg)
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    """Reject NaN and infinities at the JSON boundary.

    Raises:
        TransportError: Always, because JSON constants are unsupported.
    """
    del value
    msg = "V2 JSON must not contain non-finite numeric constants."
    raise TransportError(msg)


def _parse_json(encoded: str) -> object:
    """Return strict JSON without duplicate keys or non-finite constants.

    Raises:
        TransportError: If the input is malformed or unsupported.
    """
    try:
        return json.loads(
            encoded,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        msg = f"V2 envelope contains malformed JSON at offset {exc.pos}."
        raise TransportError(msg) from None
    except (RecursionError, ValueError) as exc:
        msg = f"V2 envelope JSON parsing failed with {safe_type_name(exc)}."
        raise TransportError(msg) from None


def _require_exact_dict(
    *,
    value: object,
    keys: frozenset[str],
    context: str,
) -> dict[str, Any]:
    """Return one exact builtin dictionary with an exact string key set.

    Raises:
        TransportError: If the type or key set is invalid.
    """
    if type(value) is not dict:
        msg = f"{context} must be an exact dict, got {safe_type_name(value)}."
        raise TransportError(msg)
    typed = cast("dict[object, Any]", value)
    raw_keys = list(dict.keys(typed))
    if any(type(key) is not str for key in raw_keys):
        msg = f"{context} keys must be exact strings."
        raise TransportError(msg)
    actual = frozenset(cast("list[str]", raw_keys))
    if actual != keys:
        msg = f"{context} must contain exactly {len(keys)} required keys."
        raise TransportError(msg)
    return cast("dict[str, Any]", typed)


def _require_string_dict(*, value: object, context: str) -> dict[str, Any]:
    """Return an exact builtin dictionary with exact string keys.

    Raises:
        TransportError: If the type or any key is invalid.
    """
    if type(value) is not dict:
        msg = f"{context} must be an exact dict, got {safe_type_name(value)}."
        raise TransportError(msg)
    typed = cast("dict[object, Any]", value)
    if any(type(key) is not str for key in dict.keys(typed)):
        msg = f"{context} keys must be exact strings."
        raise TransportError(msg)
    return cast("dict[str, Any]", typed)


def _require_int(
    *,
    value: object,
    minimum: int,
    context: str,
) -> int:
    """Return an exact integer at or above a lower bound.

    Raises:
        TransportError: If the value is not an in-range exact integer.
    """
    if type(value) is not int or value < minimum:
        msg = f"{context} must be an integer >= {minimum}."
        raise TransportError(msg)
    return value


def _require_str(*, value: object, context: str) -> str:
    """Return one exact string.

    Raises:
        TransportError: If the value is not an exact string.
    """
    if type(value) is not str:
        msg = f"{context} must be an exact string."
        raise TransportError(msg)
    return value


def _require_optional_str(*, value: object, context: str) -> str | None:
    """Return one exact string or None.

    Raises:
        TransportError: If the value has any other type.
    """
    if value is None:
        return None
    return _require_str(value=value, context=context)


def _require_list(*, value: object, context: str) -> list[object]:
    """Return one exact list.

    Raises:
        TransportError: If the value is not an exact list.
    """
    if type(value) is not list:
        msg = f"{context} must be an exact list."
        raise TransportError(msg)
    return cast("list[object]", value)


def _require_number_or_none(*, value: object, context: str) -> int | float | None:
    """Return one finite exact number or None.

    Raises:
        TransportError: If the value is boolean, nonnumeric, or non-finite.
    """
    if value is None:
        return None
    if type(value) is not int and type(value) is not float:
        msg = f"{context} must be a finite number or None."
        raise TransportError(msg)
    if type(value) is float and not math.isfinite(value):
        msg = f"{context} must be finite."
        raise TransportError(msg)
    return value


def _require_enum_string(
    *,
    value: object,
    allowed: frozenset[str],
    context: str,
) -> str:
    """Return one exact supported enum string.

    Raises:
        TransportError: If the value is not supported.
    """
    parsed = _require_str(value=value, context=context)
    if parsed not in allowed:
        msg = f"{context} is unsupported."
        raise TransportError(msg)
    return parsed


def _validate_json_tree(*, value: object, context: str) -> None:
    """Validate bounded JSON metadata without recursive Python calls.

    Raises:
        TransportError: If a value is unsupported or too deeply nested.
    """
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        current_type = type(current)
        if current_type in {type(None), bool, int, str}:
            continue
        if current_type is float:
            if not math.isfinite(cast("float", current)):
                msg = f"{context} contains a non-finite float."
                raise TransportError(msg)
            continue
        if current_type is list:
            children = cast("list[object]", current)
        elif current_type is dict:
            typed = _require_string_dict(value=current, context=context)
            children = list(typed.values())
        else:
            msg = f"{context} contains unsupported {safe_type_name(current)}."
            raise TransportError(msg)
        if depth > MAX_METADATA_DEPTH:
            msg = f"{context} exceeds the supported metadata depth."
            raise TransportError(msg)
        pending.extend((child, depth + 1) for child in children)


def _validate_metadata(*, value: object, context: str) -> dict[str, Any]:
    """Return one strictly bounded JSON metadata dictionary.

    Raises:
        TransportError: If the dictionary is malformed.
    """
    typed = _require_string_dict(value=value, context=context)
    for item in typed.values():
        _validate_json_tree(value=item, context=context)
    return typed


def _validate_eval_result(value: object) -> None:
    """Validate one evaluator-result projection.

    Raises:
        TransportError: If any field is malformed.
    """
    if value is None:
        return
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(EvalResultWire.__required_keys__),
        context="V2 EvalResult",
    )
    _require_enum_string(
        value=typed["outcome"],
        allowed=_EVAL_OUTCOME_VALUES,
        context="V2 EvalResult outcome",
    )
    _require_number_or_none(
        value=typed["confidence"],
        context="V2 EvalResult confidence",
    )
    for item in _require_list(
        value=typed["evidence"],
        context="V2 EvalResult evidence",
    ):
        _require_str(value=item, context="V2 EvalResult evidence item")
    _require_str(value=typed["rationale"], context="V2 EvalResult rationale")


def _validate_payload(value: object) -> None:
    """Validate one payload projection.

    Raises:
        TransportError: If any field is malformed.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(PayloadWire.__required_keys__),
        context="V2 Payload",
    )
    _require_str(value=typed["content"], context="V2 Payload content")
    _require_str(value=typed["id"], context="V2 Payload id")
    _require_enum_string(
        value=typed["format"],
        allowed=_PAYLOAD_FORMAT_VALUES,
        context="V2 Payload format",
    )
    _require_optional_str(value=typed["artifact"], context="V2 Payload artifact")
    _validate_metadata(value=typed["metadata"], context="V2 Payload metadata")


def _validate_request(value: object) -> None:
    """Validate one request projection.

    Raises:
        TransportError: If any field is malformed.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(RequestWire.__required_keys__),
        context="V2 Request",
    )
    _require_optional_str(value=typed["prompt"], context="V2 Request prompt")
    attachments = _require_list(
        value=typed["attachments"],
        context="V2 Request attachments",
    )
    for attachment in attachments:
        _validate_payload(attachment)


def _validate_tool_call(value: object) -> None:
    """Validate one tool-call projection.

    Raises:
        TransportError: If any field is malformed.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(ToolCallWire.__required_keys__),
        context="V2 ToolCall",
    )
    _require_str(value=typed["name"], context="V2 ToolCall name")
    _validate_metadata(value=typed["arguments"], context="V2 ToolCall arguments")
    _require_optional_str(value=typed["result"], context="V2 ToolCall result")
    _require_optional_str(
        value=typed["timestamp"],
        context="V2 ToolCall timestamp",
    )


def _validate_side_effect(value: object) -> None:
    """Validate one side-effect projection.

    Raises:
        TransportError: If any field is malformed.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(SideEffectWire.__required_keys__),
        context="V2 SideEffect",
    )
    _require_str(value=typed["kind"], context="V2 SideEffect kind")
    _validate_metadata(value=typed["details"], context="V2 SideEffect details")


def _validate_response(value: object) -> None:
    """Validate one response projection.

    Raises:
        TransportError: If any field is malformed.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(ResponseWire.__required_keys__),
        context="V2 Response",
    )
    _require_str(value=typed["text"], context="V2 Response text")
    for tool_call in _require_list(
        value=typed["tool_calls"],
        context="V2 Response tool_calls",
    ):
        _validate_tool_call(tool_call)
    for side_effect in _require_list(
        value=typed["side_effects"],
        context="V2 Response side_effects",
    ):
        _validate_side_effect(side_effect)
    _validate_metadata(value=typed["metadata"], context="V2 Response metadata")


def _validate_turn(value: object) -> None:
    """Validate one turn projection.

    Raises:
        TransportError: If any field is malformed.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(TurnWire.__required_keys__),
        context="V2 Turn",
    )
    _validate_request(typed["request"])
    _validate_response(typed["response"])
    _validate_eval_result(typed["eval_result"])
    if type(typed["turn_number"]) is not int:
        msg = "V2 Turn turn_number must be an exact integer."
        raise TransportError(msg)
    _require_optional_str(value=typed["timestamp"], context="V2 Turn timestamp")
    _require_str(
        value=typed["driver_reasoning"],
        context="V2 Turn driver_reasoning",
    )


def _validate_injection(value: object) -> None:
    """Validate one injection-record projection.

    Raises:
        TransportError: If any field is malformed.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(InjectionWire.__required_keys__),
        context="V2 InjectionRecord",
    )
    _require_optional_str(
        value=typed["payload_id"],
        context="V2 InjectionRecord payload_id",
    )
    _require_str(
        value=typed["surface_name"],
        context="V2 InjectionRecord surface_name",
    )


def _freeze_json(  # ruff: ignore[too-many-return-statements]
    value: object,
) -> object:
    """Return a type- and order-preserving hashable JSON tree.

    Raises:
        TransportError: If the value is outside the JSON projection.
    """
    value_type = type(value)
    if value_type is type(None):
        return ("none",)
    if value_type is bool:
        return ("bool", cast("bool", value))
    if value_type is int:
        return ("int", cast("int", value))
    if value_type is float:
        return ("float", cast("float", value))
    if value_type is str:
        return ("str", cast("str", value))
    if value_type is list:
        return (
            "list",
            tuple(_freeze_json(item) for item in cast("list[object]", value)),
        )
    if value_type is dict:
        typed = cast("dict[object, object]", value)
        if any(type(key) is not str for key in dict.keys(typed)):
            msg = "Canonical JSON dictionaries require exact string keys."
            raise TransportError(msg)
        items = tuple(
            (cast("str", key), _freeze_json(item))
            for key, item in sorted(
                dict.items(typed),
                key=lambda pair: cast("str", pair[0]),
            )
        )
        return ("dict", items)
    msg = f"Canonical JSON received unsupported {safe_type_name(value)}."
    raise TransportError(msg)


def _validate_result(
    *,
    value: object,
) -> tuple[Result, object]:
    """Return one strictly validated existing v1 Result projection.

    Raises:
        TransportError: If any Result field is malformed or coerced.
    """
    keys = frozenset(ResultWire.__required_keys__)
    typed = _require_exact_dict(value=value, keys=keys, context="V2 Result")
    if type(typed["safe"]) is not bool:
        msg = "V2 Result safe must be an exact boolean."
        raise TransportError(msg)
    _require_enum_string(
        value=typed["status"],
        allowed=_SAFETY_STATUS_VALUES,
        context="V2 Result status",
    )
    _require_str(value=typed["summary"], context="V2 Result summary")
    for turn in _require_list(value=typed["turns"], context="V2 Result turns"):
        _validate_turn(turn)
    _require_number_or_none(
        value=typed["duration_seconds"],
        context="V2 Result duration_seconds",
    )
    _require_optional_str(
        value=typed["harm_category"],
        context="V2 Result harm_category",
    )
    _require_str(value=typed["strategy"], context="V2 Result strategy")
    _require_enum_string(
        value=typed["observability_level"],
        allowed=_OBSERVABILITY_VALUES,
        context="V2 Result observability_level",
    )
    for injection in _require_list(
        value=typed["injections"],
        context="V2 Result injections",
    ):
        _validate_injection(injection)
    metadata = _validate_metadata(
        value=typed["metadata"],
        context="V2 Result metadata",
    )
    if _NODEID_METADATA_KEY in metadata or _SOURCE_WORKER_METADATA_KEY in metadata:
        msg = "V2 Result must not embed authoritative route metadata."
        raise TransportError(msg)
    if _RESULT_INDEX_METADATA_KEY in metadata:
        _require_int(
            value=metadata[_RESULT_INDEX_METADATA_KEY],
            minimum=0,
            context="V2 Result scheduling index",
        )
    try:
        result = _deserialize_result(data=typed)
    except WorkerOutputError as exc:
        msg = f"V2 Result validation failed: {bounded_text(str(exc))}"
        raise TransportError(msg) from None
    if result.safe is not typed["safe"]:
        msg = "V2 Result safe does not match status."
        raise TransportError(msg)
    return result, _freeze_json(typed)


def _validate_result_record(
    *,
    value: object,
) -> tuple[ValidatedResult, object]:
    """Return one strictly validated structural Result record."""
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(ResultRecordWire.__required_keys__),
        context="V2 Result record",
    )
    index = _require_int(
        value=typed["index"],
        minimum=0,
        context="V2 Result record index",
    )
    result, result_key = _validate_result(value=typed["result"])
    return (
        ValidatedResult(index=index, result=result),
        (("index", index), ("result", result_key)),
    )


def _validate_drop(value: object) -> ValidatedDrop:
    """Return one strictly validated drop record.

    Raises:
        TransportError: If the record violates the v2 schema.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(DropWire.__required_keys__),
        context="V2 drop",
    )
    index = _require_int(
        value=typed["index"],
        minimum=0,
        context="V2 drop index",
    )
    serialized_bytes = _require_int(
        value=typed["serialized_bytes"],
        minimum=1,
        context="V2 drop serialized_bytes",
    )
    if type(typed["reason"]) is not str:
        msg = "V2 drop reason must be an exact string."
        raise TransportError(msg)
    if typed["reason"] != DropReason.SIZE_LIMIT.value:
        msg = "V2 drop reason is unsupported."
        raise TransportError(msg)
    return ValidatedDrop(index=index, serialized_bytes=serialized_bytes)


def parse_envelope(  # ruff: ignore[complex-structure, too-many-locals]
    *,
    encoded: str,
    limit: int,
    nodeid: str,
) -> ValidatedEnvelope:
    """Return one parsed and strictly validated report attribute.

    Raises:
        TransportSizeError: If the attribute exceeds the configured cap.
        TransportError: If the envelope or a nested record is malformed.
    """
    del nodeid
    if len(encoded) > limit:
        msg = f"V2 report attribute exceeds its {limit}-byte cap."
        raise TransportSizeError(msg)
    try:
        encoded_size = len(encoded.encode("utf-8"))
    except UnicodeEncodeError:
        msg = "V2 report attribute is not valid UTF-8 text."
        raise TransportError(msg) from None
    if encoded_size > limit:
        msg = f"V2 report attribute is {encoded_size} bytes; cap is {limit}."
        raise TransportSizeError(msg)
    value = _parse_json(encoded)
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(EnvelopeWire.__required_keys__),
        context="V2 envelope",
    )
    if type(typed["schema_version"]) is not int:
        msg = "V2 schema_version must be an exact integer."
        raise TransportError(msg)
    if typed["schema_version"] != SCHEMA_VERSION:
        msg = "V2 envelope schema_version is unsupported."
        raise TransportError(msg)
    sequence = _require_int(
        value=typed["sequence"],
        minimum=1,
        context="V2 sequence",
    )
    produced = _require_int(
        value=typed["produced"],
        minimum=1,
        context="V2 produced",
    )
    if type(typed["results"]) is not list or type(typed["dropped"]) is not list:
        msg = "V2 results and dropped must be exact lists."
        raise TransportError(msg)
    raw_results = cast("list[object]", typed["results"])
    raw_drops = cast("list[object]", typed["dropped"])
    if produced != len(raw_results) + len(raw_drops):
        msg = "V2 produced must equal result and drop record counts."
        raise TransportError(msg)
    validated_results: list[ValidatedResult] = []
    result_keys: list[object] = []
    for raw_record in raw_results:
        validated, semantic_key = _validate_result_record(value=raw_record)
        validated_results.append(validated)
        result_keys.append(semantic_key)
    validated_drops = tuple(_validate_drop(value=drop) for drop in raw_drops)
    result_indexes = [result.index for result in validated_results]
    drop_indexes = [drop.index for drop in validated_drops]
    indexes = [*result_indexes, *drop_indexes]
    if result_indexes != sorted(result_indexes):
        msg = "V2 Result indexes must be increasing."
        raise TransportError(msg)
    if drop_indexes != sorted(drop_indexes):
        msg = "V2 drop indexes must be increasing."
        raise TransportError(msg)
    if len(indexes) != len(set(indexes)) or set(indexes) != set(range(produced)):
        msg = "V2 Result and drop indexes must partition produced records."
        raise TransportError(msg)
    semantic_key = (
        ("sequence", sequence),
        ("produced", produced),
        ("results", tuple(result_keys)),
        (
            "dropped",
            tuple((drop.index, drop.serialized_bytes) for drop in validated_drops),
        ),
    )
    return ValidatedEnvelope(
        sequence=sequence,
        produced=produced,
        results=tuple(validated_results),
        dropped=validated_drops,
        semantic_key=semantic_key,
    )


def validate_manifest(value: object) -> ValidatedManifest:
    """Return one strictly validated clean-finish manifest.

    Raises:
        TransportError: If the manifest violates the v2 schema.
    """
    typed = _require_exact_dict(
        value=value,
        keys=frozenset(ManifestWire.__required_keys__),
        context="V2 manifest",
    )
    if type(typed["schema_version"]) is not int:
        msg = "V2 manifest schema_version must be an exact integer."
        raise TransportError(msg)
    if typed["schema_version"] != SCHEMA_VERSION:
        msg = "V2 manifest schema_version is unsupported."
        raise TransportError(msg)
    envelopes_sent = _require_int(
        value=typed["envelopes_sent"],
        minimum=0,
        context="V2 manifest envelopes_sent",
    )
    last_sequence = _require_int(
        value=typed["last_sequence"],
        minimum=0,
        context="V2 manifest last_sequence",
    )
    results_sent = _require_int(
        value=typed["results_sent"],
        minimum=0,
        context="V2 manifest results_sent",
    )
    results_dropped = _require_int(
        value=typed["results_dropped"],
        minimum=0,
        context="V2 manifest results_dropped",
    )
    if envelopes_sent != last_sequence:
        msg = "V2 manifest envelopes_sent must equal last_sequence."
        raise TransportError(msg)
    semantic_key = (
        envelopes_sent,
        last_sequence,
        results_sent,
        results_dropped,
    )
    return ValidatedManifest(
        envelopes_sent=envelopes_sent,
        last_sequence=last_sequence,
        results_sent=results_sent,
        results_dropped=results_dropped,
        semantic_key=semantic_key,
    )


@dataclass(kw_only=True, eq=False)
class XdistResultTransport:
    """Per-config worker/controller runtime for authoritative v2 delivery."""

    config: pytest.Config
    session: RampartSession
    _worker: _WorkerState = field(default_factory=_WorkerState)
    _controller: _ControllerState = field(default_factory=_ControllerState)
    _is_worker: bool = field(init=False)
    _is_controller: bool = field(init=False)

    def __post_init__(self) -> None:
        self._is_worker = is_xdist_worker(config=self.config)
        self._is_controller = is_xdist_controller(config=self.config)

    def remember_results(
        self,
        *,
        item: pytest.Item,
        results: tuple[Result, ...],
    ) -> None:
        """Stash the exact absorbed tuple for this item attempt."""
        if not self._is_worker:
            return
        if ITEM_RESULTS_KEY in item.stash:
            stale = item.stash[ITEM_RESULTS_KEY]
            self._worker.results_dropped += len(stale)
            logger.warning(
                "Discarded %d stale RAMPART transport Result(s) before a rerun.",
                len(stale),
            )
        item.stash[ITEM_RESULTS_KEY] = results

    def record_source_failure(self, *, result_count: int) -> None:
        """Account for Results that could not be sourced from absorption."""
        if self._is_worker and result_count > 0:
            self._worker.results_dropped += result_count
            logger.warning(
                "RAMPART transport could not source %d absorbed Result(s).",
                result_count,
            )

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self,
        item: pytest.Item,
        call: pytest.CallInfo[None],
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        """Return a teardown report carrying one worker envelope when needed."""
        del call
        report = yield
        if not self._is_worker or report.when != "teardown":
            return report
        results = item.stash.get(ITEM_RESULTS_KEY, ())
        if ITEM_RESULTS_KEY in item.stash:
            del item.stash[ITEM_RESULTS_KEY]
        if not results:
            return report
        sequence = self._worker.last_sequence + 1
        nodeid = item.nodeid if type(item.nodeid) is str else "<invalid-nodeid>"
        try:
            built = build_envelope(
                results=results,
                nodeid=nodeid,
                sequence=sequence,
                limit=_size_limit(config=self.config),
            )
        except Exception as exc:  # ruff: ignore[blind-except] — hook isolation
            self._worker.results_dropped += len(results)
            logger.warning(
                "RAMPART transport serialization failed for %s with %s.",
                bounded_text(nodeid),
                safe_type_name(exc),
            )
            return report
        self._worker.results_dropped += built.results_dropped
        if built.encoded is None:
            return report
        setattr(report, REPORT_ATTRIBUTE, built.encoded)
        self._worker.last_sequence = sequence
        self._worker.envelopes_sent += 1
        self._worker.results_sent += built.results_sent
        return report

    def pytest_runtest_logreport(  # ruff: ignore[complex-structure, too-many-return-statements]
        self,
        report: pytest.TestReport,
    ) -> None:
        """Consume one reconstructed controller teardown attribute."""
        if not self._is_controller or report.when != "teardown":
            return
        sentinel = object()
        encoded = report.__dict__.get(REPORT_ATTRIBUTE, sentinel)
        if encoded is sentinel:
            return
        worker_id = getattr(report, "worker_id", None)
        nodeid = getattr(report, "nodeid", None)
        if type(worker_id) is not str or not worker_id:
            self._record_fault(
                code="invalid-report-worker",
                reason="v2 report has missing or invalid worker identity",
            )
            return
        self._controller.report_workers.add(worker_id)
        if type(nodeid) is not str or not nodeid:
            self._record_fault(
                code=f"invalid-report-node:{worker_id}",
                reason=f"worker {worker_id} v2 report has invalid node identity",
            )
            return
        if type(encoded) is not str or not encoded:
            self._record_fault(
                code=f"invalid-report-attribute:{worker_id}",
                reason=f"worker {worker_id} v2 report attribute is invalid",
            )
            return
        try:
            envelope = parse_envelope(
                encoded=encoded,
                limit=_size_limit(config=self.config),
                nodeid=nodeid,
            )
        except TransportError as exc:
            self._record_fault(
                code=f"malformed-envelope:{worker_id}:{nodeid}",
                reason=(
                    f"worker {worker_id} v2 envelope validation failed: "
                    f"{bounded_text(str(exc))}"
                ),
            )
            return
        except Exception as exc:  # ruff: ignore[blind-except] — hook isolation
            self._record_fault(
                code=f"envelope-boundary-error:{worker_id}:{nodeid}",
                reason=(
                    f"worker {worker_id} v2 envelope processing failed with "
                    f"{safe_type_name(exc)}"
                ),
            )
            return
        key = worker_id, envelope.sequence
        existing = self._controller.deliveries.get(key)
        if existing is not None:
            if (
                existing.nodeid == nodeid
                and existing.envelope.semantic_key == envelope.semantic_key
            ):
                return
            self._record_fault(
                code=f"conflicting-envelope:{worker_id}:{envelope.sequence}",
                reason=(
                    f"worker {worker_id} v2 sequence {envelope.sequence} "
                    "was delivered with conflicting content"
                ),
            )
            return
        self.session.append_transported_results(
            nodeid=nodeid,
            source_worker=worker_id,
            sequence=envelope.sequence,
            results=tuple(
                (validated.index, validated.result) for validated in envelope.results
            ),
        )
        self._controller.deliveries[key] = AcceptedDelivery(
            worker_id=worker_id,
            nodeid=nodeid,
            envelope=envelope,
        )
        if envelope.dropped:
            self._record_fault(
                code=f"drop-record:{worker_id}:{envelope.sequence}",
                reason=(
                    f"worker {worker_id} v2 sequence {envelope.sequence} "
                    f"dropped {len(envelope.dropped)} Result(s)"
                ),
            )

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(
        self,
        node: object,
        error: object,
    ) -> None:
        """Validate one clean worker manifest after all its report events."""
        if not self._is_controller:
            return
        worker_id = self._node_worker_id(node)
        if worker_id is None:
            self._record_fault(
                code="invalid-node-worker",
                reason="v2 node-down event has invalid worker identity",
            )
            return
        self._controller.node_down_workers.add(worker_id)
        if error is not None:
            self._record_fault(
                code=f"worker-error:{worker_id}",
                reason=f"worker {worker_id} ended without a clean v2 manifest",
            )
            return
        workeroutput = getattr(node, "workeroutput", None)
        if type(workeroutput) is not dict:
            self._record_fault(
                code=f"missing-workeroutput:{worker_id}",
                reason=f"worker {worker_id} has no valid v2 workeroutput",
            )
            return
        typed_workeroutput = cast("dict[object, object]", workeroutput)
        self._receive_manifest(
            worker_id=worker_id,
            workeroutput=typed_workeroutput,
        )
        self._receive_trial_specs(
            worker_id=worker_id,
            workeroutput=typed_workeroutput,
        )

    def _receive_manifest(
        self,
        *,
        worker_id: str,
        workeroutput: dict[object, object],
    ) -> None:
        """Validate and retain one clean worker manifest."""
        raw_manifest = workeroutput.get(MANIFEST_KEY)
        if raw_manifest is None:
            self._record_fault(
                code=f"missing-manifest:{worker_id}",
                reason=f"worker {worker_id} is missing its clean v2 manifest",
            )
            return
        try:
            manifest = validate_manifest(raw_manifest)
        except TransportError as exc:
            self._record_fault(
                code=f"malformed-manifest:{worker_id}",
                reason=(
                    f"worker {worker_id} v2 manifest validation failed: "
                    f"{bounded_text(str(exc))}"
                ),
            )
            return
        existing = self._controller.manifests.get(worker_id)
        if existing is not None and existing.semantic_key != manifest.semantic_key:
            self._record_fault(
                code=f"conflicting-manifest:{worker_id}",
                reason=f"worker {worker_id} delivered conflicting v2 manifests",
            )
            return
        self._controller.manifests[worker_id] = manifest

    def _receive_trial_specs(
        self,
        *,
        worker_id: str,
        workeroutput: dict[object, object],
    ) -> None:
        """Validate temporary clone metadata independently of Results."""
        raw_payload = workeroutput.get(TRIAL_SPECS_WORKEROUTPUT_KEY)
        if raw_payload is None:
            self._record_fault(
                code=f"missing-trial-specs:{worker_id}",
                reason=f"worker {worker_id} is missing clean trial-spec metadata",
            )
            return
        try:
            trial_specs = deserialize_trial_specs(data=raw_payload)
        except WorkerOutputError as exc:
            self._record_fault(
                code=f"malformed-trial-specs:{worker_id}",
                reason=(
                    f"worker {worker_id} trial-spec validation failed: "
                    f"{bounded_text(str(exc))}"
                ),
            )
            return
        conflicts = self.session.merge_trial_specs(trial_specs=trial_specs)
        if conflicts:
            self._record_fault(
                code=f"conflicting-trial-specs:{worker_id}",
                reason=(
                    f"worker {worker_id} delivered {len(conflicts)} "
                    "conflicting trial spec(s)"
                ),
            )

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(
        self,
        session: pytest.Session,
        exitstatus: int,
    ) -> None:
        """Publish a worker manifest or reconcile controller delivery state."""
        del exitstatus
        if self._is_worker:
            workeroutput = getattr(self.config, "workeroutput", None)
            if type(workeroutput) is dict:
                cast("dict[str, object]", workeroutput)[MANIFEST_KEY] = (
                    self._worker.manifest()
                )
            return
        if self._is_controller and session.config is self.config:
            self._reconcile_controller()

    def _validate_worker_manifest(
        self,
        *,
        worker_id: str,
        manifest: ValidatedManifest,
    ) -> None:
        deliveries = [
            delivery
            for (source_worker, _), delivery in self._controller.deliveries.items()
            if source_worker == worker_id
        ]
        sequences = sorted(delivery.envelope.sequence for delivery in deliveries)
        contiguous = len(sequences) == manifest.last_sequence and all(
            sequence == expected for expected, sequence in enumerate(sequences, start=1)
        )
        if not contiguous or len(deliveries) != manifest.envelopes_sent:
            self._record_fault(
                code=f"sequence-mismatch:{worker_id}",
                reason=(f"worker {worker_id} v2 sequence manifest does not reconcile"),
            )
        results_sent = sum(len(item.envelope.results) for item in deliveries)
        results_dropped = sum(len(item.envelope.dropped) for item in deliveries)
        if results_sent != manifest.results_sent:
            self._record_fault(
                code=f"result-count-mismatch:{worker_id}",
                reason=f"worker {worker_id} v2 Result count does not reconcile",
            )
        if results_dropped != manifest.results_dropped:
            self._record_fault(
                code=f"drop-count-mismatch:{worker_id}",
                reason=f"worker {worker_id} v2 drop count does not reconcile",
            )
        if manifest.results_dropped:
            self._record_fault(
                code=f"manifest-drops:{worker_id}",
                reason=(
                    f"worker {worker_id} v2 manifest records "
                    f"{manifest.results_dropped} dropped Result(s)"
                ),
            )

    def _reconcile_controller(self) -> None:
        if self._controller.reconciled:
            return
        self._controller.reconciled = True
        missing_node_down = (
            self._controller.report_workers - self._controller.node_down_workers
        )
        for worker_id in sorted(missing_node_down):
            self._record_fault(
                code=f"missing-node-down:{worker_id}",
                reason=f"worker {worker_id} has v2 Results but no node-down event",
            )
        for worker_id, manifest in sorted(self._controller.manifests.items()):
            self._validate_worker_manifest(worker_id=worker_id, manifest=manifest)

    def _record_fault(self, *, code: str, reason: str) -> None:
        if code in self._controller.fault_codes:
            return
        self._controller.fault_codes.add(code)
        safe_reason = bounded_text(reason)
        logger.warning("RAMPART xdist v2 transport: %s", safe_reason)
        self.session.mark_incomplete(reason=safe_reason)

    @staticmethod
    def _node_worker_id(node: object) -> str | None:
        gateway = getattr(node, "gateway", None)
        worker_id = getattr(gateway, "id", None)
        return worker_id if type(worker_id) is str and worker_id else None


RUNTIME_KEY = pytest.StashKey[XdistResultTransport]()
