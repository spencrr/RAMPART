# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from rampart.core import Result, SafetyStatus
from rampart.core.trial import (
    TRIAL_BATCH_COUNT_KEY,
    TRIAL_BATCH_ID_KEY,
    TRIAL_BATCH_INDEX_KEY,
    TRIAL_BATCH_SCHEMA,
    TRIAL_BATCH_SCHEMA_KEY,
    TRIAL_BATCH_THRESHOLD_KEY,
)
from rampart.reporting.trial_batch import _summarize_trial_batches

if TYPE_CHECKING:
    from rampart.reporting import TrialBatchSummary

_BATCH_A = "123e4567-e89b-42d3-a456-426614174000"
_BATCH_B = "123e4567-e89b-42d3-a456-426614174001"


class _HostileValue:
    def __repr__(self) -> str:
        raise RuntimeError("repr must not run")

    def __str__(self) -> str:
        raise RuntimeError("str must not run")


def _trial_result(
    *,
    status: SafetyStatus = SafetyStatus.SAFE,
    batch_id: str = _BATCH_A,
    index: int = 0,
    count: int = 1,
    threshold: float = 1.0,
    source_worker: str | None = None,
) -> Result:
    metadata: dict[str, Any] = {
        TRIAL_BATCH_SCHEMA_KEY: TRIAL_BATCH_SCHEMA,
        TRIAL_BATCH_ID_KEY: batch_id,
        TRIAL_BATCH_INDEX_KEY: index,
        TRIAL_BATCH_COUNT_KEY: count,
        TRIAL_BATCH_THRESHOLD_KEY: threshold,
    }
    if source_worker is not None:
        metadata["_rampart_source_worker"] = source_worker
    return Result(status=status, summary=f"result-{index}", metadata=metadata)


def _summaries(
    *results: Result,
) -> tuple[tuple[TrialBatchSummary, ...], tuple[str, ...]]:
    return _summarize_trial_batches(results=results)


class TestValidTrialSummaries:
    def test_counts_every_status(self) -> None:
        results = [
            _trial_result(status=status, index=index, count=4, threshold=0.25)
            for index, status in enumerate(SafetyStatus)
        ]

        summaries, diagnostics = _summaries(*results)

        [summary] = summaries
        assert diagnostics == ()
        assert summary.safe_count == 1
        assert summary.unsafe_count == 1
        assert summary.undetermined_count == 1
        assert summary.error_count == 1
        assert summary.pass_rate == pytest.approx(0.25)
        assert summary.complete is True
        assert summary.passed is True

    def test_threshold_only_allows_unsafe(self) -> None:
        results = [
            _trial_result(index=0, count=3, threshold=2 / 3),
            _trial_result(index=1, count=3, threshold=2 / 3),
            _trial_result(
                status=SafetyStatus.UNSAFE,
                index=2,
                count=3,
                threshold=2 / 3,
            ),
        ]

        summaries, _ = _summaries(*results)

        [summary] = summaries
        assert summary.unsafe_count == 1
        assert summary.passed is True

    def test_untagged_results_are_ignored(self) -> None:
        summaries, diagnostics = _summaries(
            Result(status=SafetyStatus.SAFE, summary="plain"),
        )
        assert summaries == ()
        assert diagnostics == ()

    def test_orders_interleaved_batches_by_first_occurrence(self) -> None:
        results = [
            _trial_result(batch_id=_BATCH_B, index=0, count=2),
            _trial_result(batch_id=_BATCH_A, index=0, count=2),
            _trial_result(batch_id=_BATCH_A, index=1, count=2),
            _trial_result(batch_id=_BATCH_B, index=1, count=2),
        ]

        summaries, _ = _summaries(*results)

        assert [summary.batch_id for summary in summaries] == [_BATCH_B, _BATCH_A]

    def test_same_worker_is_not_a_collision(self) -> None:
        results = [
            _trial_result(index=0, count=2, source_worker="gw0"),
            _trial_result(index=1, count=2, source_worker="gw0"),
        ]

        summaries, diagnostics = _summaries(*results)

        [summary] = summaries
        assert diagnostics == ()
        assert summary.complete is True
        assert summary.passed is True


class TestFailClosedTrialSummaries:
    def test_partial_coverage_fails_with_requested_denominator(self) -> None:
        summaries, diagnostics = _summaries(
            _trial_result(index=0, count=3, threshold=0.25),
        )

        [summary] = summaries
        assert summary.safe_count == 1
        assert summary.pass_rate == pytest.approx(1 / 3)
        assert summary.complete is False
        assert summary.passed is False
        assert "incomplete index coverage" in diagnostics[0]

    def test_duplicate_index_keeps_first_record_and_fails(self) -> None:
        summaries, diagnostics = _summaries(
            _trial_result(index=0, count=1),
            _trial_result(status=SafetyStatus.UNSAFE, index=0, count=1),
        )

        [summary] = summaries
        assert summary.safe_count == 1
        assert summary.unsafe_count == 0
        assert summary.complete is False
        assert summary.passed is False
        assert "duplicate batch indexes" in diagnostics[0]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            (TRIAL_BATCH_COUNT_KEY, 3),
            (TRIAL_BATCH_THRESHOLD_KEY, 0.5),
        ],
        ids=["count", "threshold"],
    )
    def test_inconsistent_headers_fail(self, field: str, value: object) -> None:
        first = _trial_result(index=0, count=2)
        second = _trial_result(index=1, count=2)
        second.metadata[field] = value

        summaries, diagnostics = _summaries(first, second)

        [summary] = summaries
        assert summary.complete is False
        assert summary.passed is False
        assert any("inconsistent headers" in diagnostic for diagnostic in diagnostics)

    def test_cross_worker_collision_fails(self) -> None:
        results = [
            _trial_result(index=0, count=2, source_worker="gw0"),
            _trial_result(index=1, count=2, source_worker="gw1"),
        ]

        summaries, diagnostics = _summaries(*results)

        [summary] = summaries
        assert summary.complete is False
        assert summary.passed is False
        assert any("collides across xdist workers" in item for item in diagnostics)

    def test_malformed_first_occurrence_taints_later_valid_header(self) -> None:
        malformed = _trial_result(index=0, count=1)
        malformed.metadata[TRIAL_BATCH_SCHEMA_KEY] = "rampart.trial-batch.v999"
        valid = _trial_result(index=0, count=1)

        summaries, diagnostics = _summaries(malformed, valid)

        [summary] = summaries
        assert summary.complete is False
        assert summary.passed is False
        assert "result position 0" in diagnostics[0]

    def test_noncanonical_duplicate_id_taints_canonical_batch(self) -> None:
        malformed = _trial_result(batch_id=_BATCH_A.upper())
        valid = _trial_result()

        summaries, diagnostics = _summaries(malformed, valid)

        [summary] = summaries
        assert summary.complete is False
        assert summary.passed is False
        assert any("noncanonical batch ID" in item for item in diagnostics)

    def test_unassigned_invalid_id_taints_other_summaries(self) -> None:
        malformed = _trial_result(batch_id="not-a-uuid")
        valid = _trial_result()

        summaries, diagnostics = _summaries(malformed, valid)

        [summary] = summaries
        assert summary.complete is False
        assert summary.passed is False
        assert any(
            "unassigned malformed trial metadata" in item for item in diagnostics
        )

    def test_large_partial_count_does_not_expand_requested_range(self) -> None:
        summaries, _ = _summaries(_trial_result(count=10**9))
        [summary] = summaries
        assert summary.requested_count == 10**9
        assert summary.complete is False

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            (TRIAL_BATCH_SCHEMA_KEY, "unknown"),
            (TRIAL_BATCH_COUNT_KEY, True),
            (TRIAL_BATCH_COUNT_KEY, 0),
            (TRIAL_BATCH_COUNT_KEY, "1"),
            (TRIAL_BATCH_INDEX_KEY, True),
            (TRIAL_BATCH_INDEX_KEY, 1),
            (TRIAL_BATCH_INDEX_KEY, "0"),
            (TRIAL_BATCH_THRESHOLD_KEY, True),
            (TRIAL_BATCH_THRESHOLD_KEY, 0.0),
            (TRIAL_BATCH_THRESHOLD_KEY, float("nan")),
            (TRIAL_BATCH_THRESHOLD_KEY, float("inf")),
            (TRIAL_BATCH_THRESHOLD_KEY, "1.0"),
            ("_rampart_source_worker", ""),
            ("_rampart_source_worker", 1),
        ],
    )
    def test_malformed_fields_never_create_passing_summary(
        self,
        field: str,
        value: object,
    ) -> None:
        result = _trial_result()
        result.metadata[field] = value

        summaries, diagnostics = _summaries(result)

        assert not summaries
        assert diagnostics
        assert all(len(diagnostic) < 200 for diagnostic in diagnostics)

    def test_invalid_status_never_creates_passing_summary(self) -> None:
        result = _trial_result()
        attribute = "status"
        setattr(result, attribute, "safe")

        summaries, diagnostics = _summaries(result)

        assert not summaries
        assert any("invalid Result status" in item for item in diagnostics)

    @pytest.mark.parametrize(
        "batch_id",
        [
            "not-a-uuid",
            "123e4567-e89b-12d3-a456-426614174000",
            _BATCH_A.upper(),
        ],
    )
    def test_invalid_uuid_never_creates_summary(self, batch_id: str) -> None:
        summaries, diagnostics = _summaries(_trial_result(batch_id=batch_id))
        assert summaries == ()
        assert diagnostics
        assert any("batch ID" in item for item in diagnostics)

    def test_hostile_values_are_not_rendered(self) -> None:
        result = _trial_result()
        result.metadata[TRIAL_BATCH_COUNT_KEY] = _HostileValue()

        summaries, diagnostics = _summaries(result)

        assert summaries == ()
        assert diagnostics
        assert all("\x1b" not in diagnostic for diagnostic in diagnostics)
