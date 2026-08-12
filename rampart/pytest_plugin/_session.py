# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Session-scoped state for the RAMPART pytest plugin.

Accumulates Result objects, computes deprecated clone aggregates, reconstructs
execution-domain trial summaries, and builds the final TestRunReport.
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from rampart.core.result import Result, SafetyStatus
from rampart.pytest_plugin._diagnostics import bounded_text, safe_type_name
from rampart.reporting.sink import ReportSink, TestRunReport
from rampart.reporting.trial_batch import (
    _INVALID_METADATA_CONTAINER_KEY,
    _NONCANONICAL_METADATA_CONTAINER_KEY,
    _NONCANONICAL_METADATA_KEYS_KEY,
    _extract_metadata,
    _has_trial_metadata,
    _summarize_trial_batches,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest

    from rampart.pytest_plugin._collection import ResultCollector

logger = logging.getLogger(__name__)


def _copy_result_metadata(*, metadata: dict[str, Any]) -> dict[str, Any]:
    """Copy metadata without dispatching dict-subclass mapping hooks.

    Returns:
        dict[str, Any]: Exact-dict metadata with parser provenance markers.
    """
    extracted = _extract_metadata(metadata=metadata)
    copied = cast("dict[str, Any]", extracted.values)
    if extracted.invalid_container:
        copied[_INVALID_METADATA_CONTAINER_KEY] = True
        return copied
    if extracted.has_noncanonical_keys:
        copied[_NONCANONICAL_METADATA_KEYS_KEY] = True
    if extracted.noncanonical_container and _has_trial_metadata(metadata=copied):
        copied[_NONCANONICAL_METADATA_CONTAINER_KEY] = True
    return copied


def _result_sort_key(result: Result) -> tuple[str, int, str]:
    """Return a total-ordering key for a result.

    Orders by full node ID, then the result's index within its test,
    then the originating xdist worker. The worker tie-breaker keeps the
    order total when the same node ID arrives from multiple workers
    (e.g. ``--dist=each``); it is absent — and therefore constant —
    outside xdist, so single-process ordering is unchanged.
    """
    metadata = result.metadata
    nodeid = str(
        metadata.get("_pytest_nodeid", metadata.get("_pytest_test_name", "")),
    )
    raw_index = metadata.get("_rampart_result_index", 0)
    index = raw_index if isinstance(raw_index, int) else 0
    source_worker = str(metadata.get("_rampart_source_worker", ""))
    return (nodeid, index, source_worker)


@dataclass(frozen=True, kw_only=True)
class _TransportPosition:
    """Internal structural identity for one transported Result."""

    worker_id: str
    sequence: int
    local_slot: int


@dataclass(frozen=True, kw_only=True)
class TrialSpec:
    """Trial-clone metadata captured at collection time.

    Carries the data needed to aggregate a trial group without
    depending on ``pytest.Item`` attributes — so aggregation works
    on the xdist controller, where the cloned items themselves
    may not be reachable at session finish.

    Attributes:
        base_nodeid (str): The original test's pytest node ID.
        threshold (float): Minimum pass rate required for the group.
    """

    DEFAULT_THRESHOLD: ClassVar[float] = 1.0

    base_nodeid: str
    threshold: float


@dataclass(frozen=True, kw_only=True)
class TrialGroupResult:
    """Aggregate statistics for a trial group."""

    total: int
    safe: int
    unsafe: int
    errors: int
    no_result: int
    threshold: float
    pass_rate: float
    passed: bool

    @property
    def verdict(self) -> str:
        """Human-readable verdict: PASSED or FAILED."""
        return "PASSED" if self.passed else "FAILED"

    @property
    def terminal_label(self) -> str:
        """Short label for terminal output: PASS or FAIL."""
        return "PASS" if self.passed else "FAIL"

    @property
    def detail(self) -> str:
        """Summary detail string for terminal output (e.g. '8/10 safe, 2 no-result')."""
        parts = [f"{self.safe}/{self.total} safe"]
        if self.no_result > 0:
            parts.append(f"{self.no_result} no-result")
        return ", ".join(parts)

    @property
    def has_unsafe(self) -> bool:
        """True if any trial produced an UNSAFE result."""
        return self.unsafe > 0


class RampartSession:
    """Session-scoped state for the RAMPART plugin.

    Accumulates Result objects from all tests, stores trial group
    aggregates, tracks session duration, and builds the final
    TestRunReport. Holds configured sinks for report emission.

    Args:
        sinks (list[ReportSink]): Report sinks to emit to at session
            end. Defaults to an empty list (terminal-only output).
    """

    def __init__(self, *, sinks: list[ReportSink] | None = None) -> None:
        self._results: list[Result] = []
        self._results_by_nodeid: dict[str, list[Result]] = {}
        self._trial_groups: dict[str, TrialGroupResult] = {}
        self._trial_specs: dict[str, TrialSpec] = {}
        self._sinks: list[ReportSink] = sinks or []
        self._duration_seconds: float = 0.0
        self._cached_report: TestRunReport | None = None
        self._emitted: bool = False
        self._incomplete: bool = False
        self._incomplete_reasons: list[str] = []
        self._report_metadata: dict[str, object] = {}
        self._transport_positions: dict[int, _TransportPosition] = {}
        self._append_ordinals: dict[int, int] = {}
        self._next_append_ordinal: int = 0

    @property
    def sinks(self) -> list[ReportSink]:
        """Configured report sinks."""
        return list(self._sinks)

    @property
    def results_by_nodeid(self) -> dict[str, list[Result]]:
        """Read-only view of results grouped by pytest node ID."""
        return {
            nodeid: list(results) for nodeid, results in self._results_by_nodeid.items()
        }

    @property
    def is_emitted(self) -> bool:
        """True once report emission has been attempted (idempotency guard)."""
        return self._emitted

    @property
    def is_incomplete(self) -> bool:
        """True if any worker failed to deliver complete results."""
        return self._incomplete

    @property
    def incomplete_reasons(self) -> list[str]:
        """The recorded reasons the run is incomplete (empty if complete)."""
        return list(self._incomplete_reasons)

    def add_sinks(self, *, sinks: list[ReportSink]) -> None:
        """Register additional sinks for report emission.

        Called by the fixture-based bootstrap to add team-provided
        sinks.

        Args:
            sinks (list[ReportSink]): Sinks to append.

        Raises:
            TypeError: If any item does not satisfy ReportSink.
        """
        for sink in sinks:
            if not isinstance(sink, ReportSink):
                msg = (
                    f"Expected ReportSink, got {safe_type_name(sink)}. "
                    "Sinks must implement: "
                    "async def emit_async(*, report: TestRunReport) -> None"
                )
                raise TypeError(msg)
            self._sinks.append(sink)

    def set_duration(self, *, duration_seconds: float) -> None:
        """Set the total session duration.

        Called by the plugin at session finish with the elapsed time
        since pytest_configure.

        Args:
            duration_seconds (float): Total wall-clock seconds.
        """
        self._duration_seconds = duration_seconds

    def absorb(
        self,
        *,
        node: pytest.Item,
        collector: ResultCollector,
    ) -> tuple[Result, ...]:
        """Absorb results from a completed test's collector.

        Tags each result with the short test name (extracted from the
        node ID), the full node ID, its index within the test, and the
        harm category from ``@pytest.mark.harm``. The nodeid and index
        give a total, deterministic ordering for the terminal summary and
        report regardless of xdist worker completion order.

        Results are shallow-copied before tagging to avoid mutating
        objects the test body may still reference.

        Args:
            node (pytest.Item): The test item that just completed.
            collector (ResultCollector): The test's result collector.

        Returns:
            tuple[Result, ...]: The exact tagged Result objects stored for
                this call, in collector order. The objects are shared with
                the session and must be treated as read-only.
        """
        test_name = node.nodeid.split("::")[-1] if "::" in node.nodeid else node.nodeid
        harm_marker = node.get_closest_marker("harm")
        harm_category = (
            harm_marker.args[0] if harm_marker and harm_marker.args else None
        )

        collected = collector.results
        result_index_offset = len(self._results_by_nodeid.get(node.nodeid, ()))
        tagged: list[Result] = []
        for result_index, original_result in enumerate(collected):
            # Shallow copy is sufficient because we reconstruct all
            # mutable fields we modify (currently metadata and harm_category).
            result = copy.copy(original_result)
            metadata = _copy_result_metadata(metadata=result.metadata)
            metadata["_pytest_test_name"] = test_name
            metadata["_pytest_nodeid"] = node.nodeid
            metadata["_rampart_result_index"] = result_index_offset + result_index
            result.metadata = metadata
            if harm_category is not None and result.harm_category is None:
                result.harm_category = harm_category
            tagged.append(result)
        self._append_results(nodeid=node.nodeid, results=tagged)
        return tuple(tagged)

    def append_transported_results(
        self,
        *,
        nodeid: str,
        source_worker: str,
        sequence: int,
        results: Sequence[tuple[int, Result]],
    ) -> tuple[Result, ...]:
        """Append one validated xdist envelope's successful Result records.

        Controller route metadata is applied to shallow Result copies while
        worker scheduling metadata remains untouched. Envelope sequence and
        local slots are retained only in internal ordering state.

        Args:
            nodeid (str): Authoritative node ID reconstructed by pytest-xdist.
            source_worker (str): Authoritative xdist worker identifier.
            sequence (int): Positive worker-local envelope sequence.
            results (Sequence[tuple[int, Result]]): Validated local slots and
                their Results.

        Returns:
            tuple[Result, ...]: The exact Result copies appended to the session.

        Raises:
            ValueError: If route or structural identity is invalid.
            TypeError: If a transported value is not a Result.
        """
        self._validate_transport_route(
            nodeid=nodeid,
            source_worker=source_worker,
            sequence=sequence,
        )
        tagged, positions = self._stage_transported_results(
            nodeid=nodeid,
            source_worker=source_worker,
            sequence=sequence,
            results=results,
        )
        self._append_results(
            nodeid=nodeid,
            results=tagged,
            transport_positions=positions,
        )
        return tuple(tagged)

    def record_trial_group(
        self,
        *,
        base_nodeid: str,
        clone_nodeids: Sequence[str],
        threshold: float,
    ) -> None:
        """Record aggregate statistics for a trial group.

        Semantics:
        - Any UNSAFE result across all trials -> group FAILS
        - threshold is the minimum pass rate (SAFE / total).
          e.g. 0.8 means at least 80% of runs must be SAFE.
        - ERROR results count against the pass rate (they're not SAFE).
        - Clones with zero results (skipped or crashed before producing
          a Result) are tracked as ``no_result`` and count against
          the pass rate.

        Args:
            base_nodeid (str): The original test's node ID.
            clone_nodeids (Sequence[str]): Pytest node IDs of all clones
                in this trial group.
            threshold (float): Minimum pass rate required.
        """
        if not clone_nodeids:
            return

        # Trial gates remain per logical clone; raw reports retain every Result.
        total = len(clone_nodeids)
        unsafe_count = 0
        error_count = 0
        safe_count = 0
        no_result_count = 0

        for nodeid in clone_nodeids:
            node_results = self._results_by_nodeid.get(nodeid, [])
            if not node_results:
                no_result_count += 1
                continue
            has_unsafe = any(r.status == SafetyStatus.UNSAFE for r in node_results)
            has_error = any(r.status == SafetyStatus.ERROR for r in node_results)
            has_safe = any(r.status == SafetyStatus.SAFE for r in node_results)
            if has_unsafe:
                unsafe_count += 1
            elif has_error:
                error_count += 1
            elif has_safe:
                safe_count += 1

        pass_rate = safe_count / total if total > 0 else 0.0
        passed = unsafe_count == 0 and pass_rate >= threshold

        self._trial_groups[base_nodeid] = TrialGroupResult(
            total=total,
            safe=safe_count,
            unsafe=unsafe_count,
            errors=error_count,
            no_result=no_result_count,
            threshold=threshold,
            pass_rate=pass_rate,
            passed=passed,
        )

    def register_trial_spec(
        self,
        *,
        clone_nodeid: str,
        base_nodeid: str,
        threshold: float,
    ) -> None:
        """Record trial metadata for a cloned item at collection time.

        Called from ``pytest_collection_modifyitems`` whenever a
        ``@pytest.mark.trial`` test is expanded into clones. Stores
        the data needed for session-end aggregation in a form that
        survives the xdist worker→controller boundary.

        Identical re-registration (same key, same spec) is a no-op so
        that repeated collection passes (e.g., in workers and the
        controller) converge safely.

        Args:
            clone_nodeid (str): Node ID of the cloned item.
            base_nodeid (str): Node ID of the original (uncloned) item.
            threshold (float): Pass-rate threshold from the trial marker.
        """
        self._trial_specs[clone_nodeid] = TrialSpec(
            base_nodeid=base_nodeid,
            threshold=threshold,
        )

    def merge_trial_specs(
        self,
        *,
        trial_specs: Mapping[str, TrialSpec],
    ) -> tuple[str, ...]:
        """Merge trial specs received from an xdist worker payload.

        Idempotent: re-merging identical specs is a no-op. Spec values
        from workers should match because the same plugin code runs in every
        process. Conflicting values never replace the first trusted spec.

        Args:
            trial_specs (Mapping[str, TrialSpec]): Specs keyed by
                clone node ID.

        Returns:
            tuple[str, ...]: Clone node IDs whose specs conflicted.
        """
        conflicts: list[str] = []
        for clone_nodeid, spec in trial_specs.items():
            existing = self._trial_specs.get(clone_nodeid)
            if existing is None:
                self._trial_specs[clone_nodeid] = spec
            elif existing != spec:
                conflicts.append(clone_nodeid)
        return tuple(conflicts)

    @property
    def has_results(self) -> bool:
        """True if any results have been collected."""
        return bool(self._results)

    @property
    def trial_groups(self) -> dict[str, TrialGroupResult]:
        """Trial group aggregates, keyed by base node ID."""
        return dict(self._trial_groups)

    @property
    def trial_specs(self) -> dict[str, TrialSpec]:
        """Read-only view of registered trial specs, keyed by clone node ID."""
        return dict(self._trial_specs)

    def mark_emitted(self) -> None:
        """Mark the session as having attempted report emission."""
        self._emitted = True

    def mark_incomplete(self, *, reason: str) -> None:
        """Record that a worker failed to deliver complete results.

        Args:
            reason (str): A short human-readable explanation surfaced
                in the report metadata.
        """
        self._incomplete = True
        self._incomplete_reasons.append(reason)
        self._cached_report = None

    def set_report_metadata(self, *, metadata: dict[str, object]) -> None:
        """Attach run-level metadata that will appear on ``TestRunReport``.

        Used by the plugin to surface xdist run-mode information
        (active, worker count, dist mode). Subsequent calls merge into
        existing metadata.

        Args:
            metadata (dict[str, object]): Key/value pairs to attach.
        """
        self._report_metadata.update(metadata)
        self._cached_report = None

    def build_report(self) -> TestRunReport:
        """Build a TestRunReport from all collected results.

        The report is cached and reused on subsequent calls. The
        cache is invalidated when new results are absorbed or merged
        or when metadata is updated.

        Results are sorted by ``(_pytest_nodeid, _rampart_result_index,
        _rampart_source_worker)`` plus private envelope sequence/local-slot
        tie breakers for a total, deterministic ordering across xdist arrival
        orders. ``_pytest_nodeid`` falls back to ``_pytest_test_name`` and
        transport positions are absent outside xdist, so single-process
        ordering is unaffected.

        These leading-underscore keys are RAMPART scheduling bookkeeping,
        namespaced to avoid colliding with user-supplied result metadata.

        Returns:
            TestRunReport: Aggregated test run results.
        """
        if self._cached_report is not None:
            return self._cached_report
        sorted_results = sorted(self._results, key=self._stored_result_sort_key)
        trial_batches, trial_diagnostics = _summarize_trial_batches(
            results=sorted_results,
        )
        for diagnostic in trial_diagnostics:
            logger.warning(
                "Invalid RAMPART trial batch metadata: %s",
                bounded_text(diagnostic),
            )
        counts = Counter(r.status for r in sorted_results)
        metadata: dict[str, Any] = dict(self._report_metadata)
        if self._incomplete:
            metadata["incomplete"] = True
            metadata["incomplete_reasons"] = list(self._incomplete_reasons)
        self._cached_report = TestRunReport(
            results=sorted_results,
            total_runs=len(sorted_results),
            passed=counts[SafetyStatus.SAFE],
            failed=counts[SafetyStatus.UNSAFE],
            undetermined=counts[SafetyStatus.UNDETERMINED],
            errors=counts[SafetyStatus.ERROR],
            duration_seconds=self._duration_seconds,
            metadata=metadata,
            trial_batches=trial_batches,
        )
        return self._cached_report

    def _append_results(
        self,
        *,
        nodeid: str,
        results: Sequence[Result],
        transport_positions: Sequence[_TransportPosition] | None = None,
    ) -> None:
        """Append staged Results to every session view and invalidate caches.

        Raises:
            ValueError: If transport positions do not align with Results.
        """
        staged_results = list(results)
        positions = (
            [None] * len(staged_results)
            if transport_positions is None
            else list(transport_positions)
        )
        if len(positions) != len(staged_results):
            msg = "Transport positions must align one-to-one with Results."
            raise ValueError(msg)
        start = self._next_append_ordinal
        ordinals = {
            id(result): start + offset for offset, result in enumerate(staged_results)
        }
        positioned = {
            id(result): position
            for result, position in zip(staged_results, positions, strict=True)
            if position is not None
        }
        node_results = self._results_by_nodeid.setdefault(nodeid, [])
        self._results.extend(staged_results)
        node_results.extend(staged_results)
        self._append_ordinals.update(ordinals)
        self._transport_positions.update(positioned)
        self._next_append_ordinal += len(staged_results)
        self._cached_report = None

    def _stored_result_sort_key(
        self,
        result: Result,
    ) -> tuple[str, int, str, int, int, int]:
        """Return deterministic report ordering with private route tie breakers."""
        position = self._transport_positions.get(id(result))
        sequence = position.sequence if position is not None else -1
        local_slot = position.local_slot if position is not None else -1
        ordinal = self._append_ordinals.get(id(result), -1)
        return (*_result_sort_key(result), sequence, local_slot, ordinal)

    @staticmethod
    def _validate_transport_route(
        *,
        nodeid: str,
        source_worker: str,
        sequence: int,
    ) -> None:
        """Validate authoritative route identity before staging Results.

        Raises:
            ValueError: If any route component is invalid.
        """
        if type(nodeid) is not str or not nodeid:
            msg = "Transported Result nodeid must be a nonempty exact string."
            raise ValueError(msg)
        if type(source_worker) is not str or not source_worker:
            msg = "Transported Result worker must be a nonempty exact string."
            raise ValueError(msg)
        if type(sequence) is not int or sequence < 1:
            msg = "Transported Result sequence must be an integer >= 1."
            raise ValueError(msg)

    @staticmethod
    def _stage_transported_results(
        *,
        nodeid: str,
        source_worker: str,
        sequence: int,
        results: Sequence[tuple[int, Result]],
    ) -> tuple[list[Result], list[_TransportPosition]]:
        """Copy, route-tag, and structurally validate transported Results.

        Returns:
            tuple[list[Result], list[_TransportPosition]]: Staged Result copies
                and their aligned private transport positions.

        Raises:
            TypeError: If any value is not a Result.
            ValueError: If local slots are invalid.
        """
        tagged: list[Result] = []
        positions: list[_TransportPosition] = []
        previous_slot = -1
        for local_slot, original_result in results:
            if type(local_slot) is not int or local_slot <= previous_slot:
                msg = "Transported Result local slots must be increasing integers."
                raise ValueError(msg)
            if not isinstance(original_result, Result):
                msg = "Transported values must be Result instances."
                raise TypeError(msg)
            result = copy.copy(original_result)
            result.metadata = {
                **result.metadata,
                "_pytest_nodeid": nodeid,
                "_rampart_source_worker": source_worker,
            }
            tagged.append(result)
            positions.append(
                _TransportPosition(
                    worker_id=source_worker,
                    sequence=sequence,
                    local_slot=local_slot,
                ),
            )
            previous_slot = local_slot
        return tagged, positions
