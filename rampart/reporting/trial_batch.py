# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Trial batch summaries reconstructed from collected Result metadata."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from rampart.core import SafetyStatus
from rampart.core.trial import (
    TRIAL_BATCH_COUNT_KEY,
    TRIAL_BATCH_ID_KEY,
    TRIAL_BATCH_INDEX_KEY,
    TRIAL_BATCH_SCHEMA,
    TRIAL_BATCH_SCHEMA_KEY,
    TRIAL_BATCH_THRESHOLD_KEY,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rampart.core import Result

_SOURCE_WORKER_KEY = "_rampart_source_worker"
_INVALID_METADATA_CONTAINER_KEY = "_rampart_trial_batch_invalid_metadata"
_NONCANONICAL_METADATA_CONTAINER_KEY = "_rampart_trial_batch_noncanonical_metadata"
_UUID4_VERSION = 4
_TRIAL_BATCH_KEYS = (
    TRIAL_BATCH_SCHEMA_KEY,
    TRIAL_BATCH_ID_KEY,
    TRIAL_BATCH_INDEX_KEY,
    TRIAL_BATCH_COUNT_KEY,
    TRIAL_BATCH_THRESHOLD_KEY,
)

__all__ = ["TrialBatchSummary"]


@dataclass(frozen=True, kw_only=True)
class TrialBatchSummary:
    """Aggregate reporting projection for one represented trial batch."""

    batch_id: str
    requested_count: int
    threshold: float
    safe_count: int
    unsafe_count: int
    undetermined_count: int
    error_count: int
    pass_rate: float
    complete: bool
    passed: bool
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class _TrialRecord:
    """Validated trial metadata from one collected Result."""

    batch_id: str
    index: int
    count: int
    threshold: float
    status: SafetyStatus
    source_worker: str | None


@dataclass(frozen=True, kw_only=True)
class _ParsedTrial:
    """Parsed trial state for one collected Result."""

    batch_id: str | None
    record: _TrialRecord | None
    diagnostics: tuple[str, ...]
    has_unassigned_metadata: bool


@dataclass(kw_only=True)
class _BatchGroup:
    """Mutable parsing state for one canonical batch ID."""

    batch_id: str
    records: list[_TrialRecord] = field(default_factory=list[_TrialRecord])
    diagnostics: list[str] = field(default_factory=list[str])


def _summarize_trial_batches(
    *,
    results: Sequence[Result],
) -> tuple[tuple[TrialBatchSummary, ...], tuple[str, ...]]:
    """Reconstruct trial summaries without mutating or dropping Results.

    Args:
        results (Sequence[Result]): Results in final deterministic report order.

    Returns:
        tuple[tuple[TrialBatchSummary, ...], tuple[str, ...]]: Ordered summaries
            and bounded diagnostic messages.
    """
    groups: dict[str, _BatchGroup] = {}
    diagnostics: list[str] = []
    has_unassigned_metadata = False

    for position, result in enumerate(results):
        parsed = _parse_trial(result=result, position=position)
        if parsed.batch_id is None:
            diagnostics.extend(parsed.diagnostics)
            has_unassigned_metadata |= parsed.has_unassigned_metadata
            continue
        group = groups.setdefault(
            parsed.batch_id,
            _BatchGroup(batch_id=parsed.batch_id),
        )
        group.diagnostics.extend(parsed.diagnostics)
        if parsed.record is not None:
            group.records.append(parsed.record)

    summaries: list[TrialBatchSummary] = []
    for group in groups.values():
        if has_unassigned_metadata:
            group.diagnostics.append(
                _batch_diagnostic(
                    batch_id=group.batch_id,
                    reason="is tainted by unassigned malformed trial metadata",
                ),
            )
        summary, group_diagnostics = _summarize_group(group=group)
        diagnostics.extend(group_diagnostics)
        if summary is not None:
            summaries.append(summary)
    return tuple(summaries), tuple(diagnostics)


def _parse_trial(*, result: Result, position: int) -> _ParsedTrial:
    """Parse trial state without trusting the Result metadata container.

    Returns:
        _ParsedTrial: Validated record, diagnostics, and assignment state.
    """
    metadata = result.metadata
    if not issubclass(type(metadata), dict):
        return _ParsedTrial(
            batch_id=None,
            record=None,
            diagnostics=(
                _position_diagnostic(
                    position=position,
                    reason="has an invalid metadata container",
                ),
            ),
            has_unassigned_metadata=True,
        )
    if _metadata_has_key(metadata=metadata, key=_INVALID_METADATA_CONTAINER_KEY):
        return _ParsedTrial(
            batch_id=None,
            record=None,
            diagnostics=(
                _position_diagnostic(
                    position=position,
                    reason="has an invalid metadata container",
                ),
            ),
            has_unassigned_metadata=True,
        )
    if not _has_trial_metadata(metadata=metadata):
        return _ParsedTrial(
            batch_id=None,
            record=None,
            diagnostics=(),
            has_unassigned_metadata=False,
        )

    issues: list[str] = []
    if type(metadata) is not dict or _metadata_has_key(
        metadata=metadata,
        key=_NONCANONICAL_METADATA_CONTAINER_KEY,
    ):
        issues.append(
            _position_diagnostic(
                position=position,
                reason="has a noncanonical metadata container",
            ),
        )
    batch_id, canonical_id = _parse_batch_id(metadata=metadata)
    if batch_id is None:
        issues.append(
            _position_diagnostic(
                position=position,
                reason="has an invalid batch ID",
            ),
        )
        return _ParsedTrial(
            batch_id=None,
            record=None,
            diagnostics=tuple(issues),
            has_unassigned_metadata=True,
        )
    if not canonical_id:
        issues.append(
            _position_diagnostic(
                position=position,
                reason="has a noncanonical batch ID",
            ),
        )
        return _ParsedTrial(
            batch_id=batch_id,
            record=None,
            diagnostics=tuple(issues),
            has_unassigned_metadata=False,
        )

    record, issue = _parse_trial_record(
        result=result,
        metadata=metadata,
        batch_id=batch_id,
        position=position,
    )
    if issue is not None:
        issues.append(issue)
    return _ParsedTrial(
        batch_id=batch_id,
        record=record,
        diagnostics=tuple(issues),
        has_unassigned_metadata=False,
    )


def _has_trial_metadata(*, metadata: dict[str, object]) -> bool:
    """Return whether underlying dict storage contains any trial key."""
    return any(
        _metadata_has_key(metadata=metadata, key=key) for key in _TRIAL_BATCH_KEYS
    )


def _metadata_has_key(*, metadata: dict[str, object], key: str) -> bool:
    """Read key presence directly from underlying dict storage.

    Returns:
        bool: Whether the underlying dict storage contains the key.
    """
    return dict.__contains__(  # ruff: ignore[unnecessary-dunder-call]
        metadata,
        key,
    )


def _parse_batch_id(*, metadata: dict[str, object]) -> tuple[str | None, bool]:
    """Return a normalized UUID4 batch ID and whether its input was canonical."""
    value = dict.get(metadata, TRIAL_BATCH_ID_KEY)
    if type(value) is not str:
        return None, False
    try:
        parsed = UUID(value)
    except ValueError:
        return None, False
    if parsed.version != _UUID4_VERSION:
        return None, False
    canonical = str(parsed)
    return canonical, canonical == value


def _parse_trial_record(  # ruff: ignore[too-many-return-statements]
    *,
    result: Result,
    metadata: dict[str, object],
    batch_id: str,
    position: int,
) -> tuple[_TrialRecord | None, str | None]:
    """Validate one tagged Result and return its normalized record.

    Returns:
        tuple[_TrialRecord | None, str | None]: The record or its diagnostic.
    """
    schema = dict.get(metadata, TRIAL_BATCH_SCHEMA_KEY)
    if type(schema) is not str or schema != TRIAL_BATCH_SCHEMA:
        return None, _position_diagnostic(
            position=position,
            reason="has an unknown or invalid schema",
        )
    count = _parse_count(metadata=metadata)
    if count is None:
        return None, _position_diagnostic(
            position=position,
            reason="has an invalid requested count",
        )
    index = _parse_index(metadata=metadata, count=count)
    if index is None:
        return None, _position_diagnostic(
            position=position,
            reason="has an invalid batch index",
        )
    threshold = _parse_threshold(metadata=metadata)
    if threshold is None:
        return None, _position_diagnostic(
            position=position,
            reason="has an invalid threshold",
        )
    if type(result.status) is not SafetyStatus:
        return None, _position_diagnostic(
            position=position,
            reason="has an invalid Result status",
        )
    source_worker, valid_source = _parse_source_worker(metadata=metadata)
    if not valid_source:
        return None, _position_diagnostic(
            position=position,
            reason="has an invalid source worker",
        )
    return (
        _TrialRecord(
            batch_id=batch_id,
            index=index,
            count=count,
            threshold=threshold,
            status=result.status,
            source_worker=source_worker,
        ),
        None,
    )


def _parse_count(*, metadata: dict[str, object]) -> int | None:
    """Return a positive exact integer count."""
    value = dict.get(metadata, TRIAL_BATCH_COUNT_KEY)
    if type(value) is not int or value <= 0:
        return None
    return value


def _parse_index(*, metadata: dict[str, object], count: int) -> int | None:
    """Return an exact integer index inside the requested range."""
    value = dict.get(metadata, TRIAL_BATCH_INDEX_KEY)
    if type(value) is not int or not 0 <= value < count:
        return None
    return value


def _parse_threshold(*, metadata: dict[str, object]) -> float | None:
    """Return a finite exact numeric threshold in ``(0, 1]``."""
    value = dict.get(metadata, TRIAL_BATCH_THRESHOLD_KEY)
    if type(value) is not int and type(value) is not float:
        return None
    try:
        threshold = float(value)
    except OverflowError:
        return None
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        return None
    return threshold


def _parse_source_worker(
    *,
    metadata: dict[str, object],
) -> tuple[str | None, bool]:
    """Return an optional valid xdist source-worker identifier."""
    if not _metadata_has_key(metadata=metadata, key=_SOURCE_WORKER_KEY):
        return None, True
    value = dict.get(metadata, _SOURCE_WORKER_KEY)
    if type(value) is not str or not value:
        return None, False
    return value, True


def _summarize_group(
    *,
    group: _BatchGroup,
) -> tuple[TrialBatchSummary | None, tuple[str, ...]]:
    """Build one fail-closed summary from a parsed batch group.

    Returns:
        tuple[TrialBatchSummary | None, tuple[str, ...]]: The optional summary
            and its diagnostics.
    """
    diagnostics = list(group.diagnostics)
    if not group.records:
        diagnostics.append(
            _batch_diagnostic(
                batch_id=group.batch_id,
                reason="has no valid metadata header",
            ),
        )
        return None, tuple(diagnostics)

    canonical = group.records[0]
    indexed: dict[int, _TrialRecord] = {}
    sources: set[str | None] = set()
    for record in group.records:
        sources.add(record.source_worker)
        if not _headers_match(record=record, canonical=canonical):
            _append_unique(
                values=diagnostics,
                value=_batch_diagnostic(
                    batch_id=group.batch_id,
                    reason="has inconsistent headers",
                ),
            )
            continue
        if record.index in indexed:
            _append_unique(
                values=diagnostics,
                value=_batch_diagnostic(
                    batch_id=group.batch_id,
                    reason="contains duplicate batch indexes",
                ),
            )
            continue
        indexed[record.index] = record

    if len(sources) > 1:
        _append_unique(
            values=diagnostics,
            value=_batch_diagnostic(
                batch_id=group.batch_id,
                reason="collides across xdist workers",
            ),
        )
    if len(indexed) != canonical.count:
        _append_unique(
            values=diagnostics,
            value=_batch_diagnostic(
                batch_id=group.batch_id,
                reason="has incomplete index coverage",
            ),
        )

    counts = Counter(record.status for record in indexed.values())
    pass_rate = counts[SafetyStatus.SAFE] / canonical.count
    complete = not diagnostics
    summary = TrialBatchSummary(
        batch_id=group.batch_id,
        requested_count=canonical.count,
        threshold=canonical.threshold,
        safe_count=counts[SafetyStatus.SAFE],
        unsafe_count=counts[SafetyStatus.UNSAFE],
        undetermined_count=counts[SafetyStatus.UNDETERMINED],
        error_count=counts[SafetyStatus.ERROR],
        pass_rate=pass_rate,
        complete=complete,
        passed=complete and pass_rate >= canonical.threshold,
        diagnostics=tuple(diagnostics),
    )
    return summary, tuple(diagnostics)


def _headers_match(*, record: _TrialRecord, canonical: _TrialRecord) -> bool:
    """Return whether two records share count and threshold headers."""
    return (
        record.batch_id == canonical.batch_id
        and record.count == canonical.count
        and record.threshold == canonical.threshold
    )


def _position_diagnostic(*, position: int, reason: str) -> str:
    """Build a bounded diagnostic for one result position.

    Returns:
        str: The diagnostic.
    """
    return f"Trial batch metadata at result position {position} {reason}."


def _batch_diagnostic(*, batch_id: str, reason: str) -> str:
    """Build a bounded diagnostic for one validated batch ID.

    Returns:
        str: The diagnostic.
    """
    return f"Trial batch {batch_id} {reason}."


def _append_unique(*, values: list[str], value: str) -> None:
    """Append a diagnostic only once."""
    if value not in values:
        values.append(value)
