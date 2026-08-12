# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Execution-domain trial batching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from rampart.core.execution import BaseExecution
from rampart.core.result import Result, SafetyStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from rampart.core.adapter import AgentAdapter

_UUID4_VERSION = 4

TRIAL_BATCH_SCHEMA: str = "rampart.trial-batch.v1"
TRIAL_BATCH_SCHEMA_KEY: str = "_rampart_trial_batch_schema"
TRIAL_BATCH_ID_KEY: str = "_rampart_trial_batch_id"
TRIAL_BATCH_INDEX_KEY: str = "_rampart_trial_batch_index"
TRIAL_BATCH_COUNT_KEY: str = "_rampart_trial_batch_count"
TRIAL_BATCH_THRESHOLD_KEY: str = "_rampart_trial_batch_threshold"

__all__ = [
    "TrialBatch",
    "execute_trials_async",
]


@dataclass(frozen=True, kw_only=True)
class TrialBatch:
    """Aggregate outcome of one complete execution-domain trial batch.

    The outer batch is immutable, but its ``results`` are the original mutable
    Result objects returned by each execution. No Result is copied.

    Args:
        batch_id (str): Canonical UUID4 identifier shared by all batch Results.
        results (tuple[Result, ...]): Ordered original Results.
        requested_count (int): Number of requested executions.
        threshold (float): Minimum SAFE fraction required to pass.
    """

    batch_id: str
    results: tuple[Result, ...]
    requested_count: int
    threshold: float = 1.0

    def __post_init__(self) -> None:
        """Validate batch inputs.

        Raises:
            TypeError: If Results are not provided as a tuple of Result objects.
            ValueError: If the ID, count, threshold, length, identity, or status
                is invalid.
        """
        if not _is_canonical_uuid4(self.batch_id):
            msg = "batch_id must be a canonical UUID4 string."
            raise ValueError(msg)
        count = _validate_count(self.requested_count)
        threshold = _validate_threshold(self.threshold)
        if type(self.results) is not tuple:
            msg = "results must be a tuple of Result objects."
            raise TypeError(msg)
        if len(self.results) != count:
            msg = "results length must equal requested_count."
            raise ValueError(msg)
        if not all(isinstance(result, Result) for result in self.results):
            msg = "results must contain only Result objects."
            raise TypeError(msg)
        if not all(type(result.status) is SafetyStatus for result in self.results):
            msg = "Every Result status must be a SafetyStatus."
            raise ValueError(msg)
        previous: list[Result] = []
        for result in self.results:
            _validate_result_identity(result=result, previous=previous)
            previous.append(result)

        object.__setattr__(self, "threshold", threshold)

    @property
    def safe_count(self) -> int:
        """Current number of SAFE Results."""
        return self._count_status(SafetyStatus.SAFE)

    @property
    def unsafe_count(self) -> int:
        """Current number of UNSAFE Results."""
        return self._count_status(SafetyStatus.UNSAFE)

    @property
    def undetermined_count(self) -> int:
        """Current number of UNDETERMINED Results."""
        return self._count_status(SafetyStatus.UNDETERMINED)

    @property
    def error_count(self) -> int:
        """Current number of ERROR Results."""
        return self._count_status(SafetyStatus.ERROR)

    @property
    def pass_rate(self) -> float:
        """Current SAFE fraction over the requested count."""
        return self.safe_count / self.requested_count

    @property
    def passed(self) -> bool:
        """Whether the current pass rate meets the threshold."""
        return self._has_valid_statuses() and self.pass_rate >= self.threshold

    def __bool__(self) -> bool:
        """Return whether the batch met its threshold."""
        return self.passed

    def __repr__(self) -> str:
        """Return aggregate fields without exposing Result evidence."""
        return (
            f"TrialBatch(batch_id={self.batch_id!r}, "
            f"requested_count={self.requested_count}, "
            f"threshold={self.threshold!r}, "
            f"safe_count={self.safe_count}, "
            f"unsafe_count={self.unsafe_count}, "
            f"undetermined_count={self.undetermined_count}, "
            f"error_count={self.error_count}, "
            f"pass_rate={self.pass_rate!r}, "
            f"passed={self.passed})"
        )

    def _count_status(self, status: SafetyStatus) -> int:
        """Return the number of current Results with one exact status."""
        return sum(result.status is status for result in self.results)

    def _has_valid_statuses(self) -> bool:
        """Return whether every current Result has one exact SafetyStatus."""
        return all(type(result.status) is SafetyStatus for result in self.results)


async def execute_trials_async(
    *,
    execution_factory: Callable[[], BaseExecution],
    adapter: AgentAdapter,
    count: int,
    threshold: float = 1.0,
) -> TrialBatch:
    """Execute fresh execution objects sequentially as one trial batch.

    Each execution receives the exact same adapter. Results are tagged after
    their ``ON_POST_EXECUTE`` handlers complete, so the metadata is intended for
    post-execution reporting rather than lifecycle-handler consumption.

    Args:
        execution_factory (Callable[[], BaseExecution]): Factory returning a
            fresh execution for every trial.
        adapter (AgentAdapter): Shared adapter passed to every execution.
        count (int): Positive number of executions to run.
        threshold (float): Minimum SAFE fraction required to pass. Defaults to
            1.0.

    Returns:
        TrialBatch: Immutable aggregate containing the original ordered Results.

    Raises:
        TypeError: If validation fails or the factory returns a non-execution.
        ValueError: If validation fails or an execution/Result identity is reused.
    """
    requested_count = _validate_count(count)
    normalized_threshold = _validate_threshold(threshold)
    batch_id = str(uuid4())
    executions: list[BaseExecution] = []
    results: list[Result] = []

    for index in range(requested_count):
        execution = execution_factory()
        _validate_execution(execution=execution, previous=executions)
        executions.append(execution)

        result = _validate_trial_result(
            await execution.execute_async(adapter=adapter),
        )
        _validate_result_identity(result=result, previous=results)
        _tag_result(
            result=result,
            batch_id=batch_id,
            index=index,
            count=requested_count,
            threshold=normalized_threshold,
        )
        results.append(result)

    return TrialBatch(
        batch_id=batch_id,
        results=tuple(results),
        requested_count=requested_count,
        threshold=normalized_threshold,
    )


def _validate_count(value: object) -> int:
    """Return a validated positive, non-bool integer.

    Raises:
        TypeError: If the value is not an integer or is bool.
        ValueError: If the value is not positive.
    """
    if type(value) is not int:
        msg = "count must be a positive integer and must not be bool."
        raise TypeError(msg)
    if value <= 0:
        msg = "count must be greater than zero."
        raise ValueError(msg)
    return value


def _validate_threshold(value: object) -> float:
    """Return a finite, non-bool threshold in ``(0, 1]``.

    Raises:
        TypeError: If the value is not numeric or is bool.
        ValueError: If the value is non-finite or outside ``(0, 1]``.
    """
    if type(value) is not int and type(value) is not float:
        msg = "threshold must be a finite number in (0, 1] and must not be bool."
        raise TypeError(msg)
    try:
        threshold = float(value)
    except OverflowError:
        msg = "threshold must be a finite number in (0, 1]."
        raise ValueError(msg) from None
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        msg = "threshold must be a finite number in (0, 1]."
        raise ValueError(msg)
    return threshold


def _validate_execution(
    *,
    execution: object,
    previous: list[BaseExecution],
) -> None:
    """Require a new BaseExecution identity for the next trial.

    Raises:
        TypeError: If the factory product is not a BaseExecution.
        ValueError: If the execution identity was already used.
    """
    if not isinstance(execution, BaseExecution):
        msg = "execution_factory must return a BaseExecution."
        raise TypeError(msg)
    if any(execution is prior for prior in previous):
        msg = "execution_factory must return a distinct BaseExecution for every trial."
        raise ValueError(msg)


def _validate_trial_result(value: object) -> Result:
    """Return one valid Result from a trial execution.

    Raises:
        TypeError: If the execution did not return a Result.
        ValueError: If the Result status is not an exact SafetyStatus.
    """
    if not isinstance(value, Result):
        msg = "Every trial execution must return a Result."
        raise TypeError(msg)
    if type(value.status) is not SafetyStatus:
        msg = "Every trial execution Result status must be a SafetyStatus."
        raise ValueError(msg)
    return value


def _validate_result_identity(*, result: Result, previous: list[Result]) -> None:
    """Reject a Result object already returned by an earlier trial.

    Raises:
        ValueError: If the Result identity was already returned.
    """
    if any(result is prior for prior in previous):
        msg = "Trial batches require distinct Result object identities."
        raise ValueError(msg)


def _tag_result(
    *,
    result: Result,
    batch_id: str,
    index: int,
    count: int,
    threshold: float,
) -> None:
    """Atomically merge authoritative trial metadata into a Result."""
    result.metadata = {
        **result.metadata,
        TRIAL_BATCH_SCHEMA_KEY: TRIAL_BATCH_SCHEMA,
        TRIAL_BATCH_ID_KEY: batch_id,
        TRIAL_BATCH_INDEX_KEY: index,
        TRIAL_BATCH_COUNT_KEY: count,
        TRIAL_BATCH_THRESHOLD_KEY: threshold,
    }


def _is_canonical_uuid4(value: object) -> bool:
    """Return whether a value is a canonical lowercase UUID4 string."""
    if type(value) is not str:
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == _UUID4_VERSION and str(parsed) == value
