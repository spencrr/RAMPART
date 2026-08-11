# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Self
from uuid import UUID

import pytest

from rampart import TrialBatch, execute_trials_async
from rampart.core import (
    TRIAL_BATCH_COUNT_KEY,
    TRIAL_BATCH_ID_KEY,
    TRIAL_BATCH_INDEX_KEY,
    TRIAL_BATCH_SCHEMA,
    TRIAL_BATCH_SCHEMA_KEY,
    TRIAL_BATCH_THRESHOLD_KEY,
    AgentAdapter,
    AppManifest,
    BaseExecution,
    ExecutionEvent,
    ExecutionEventData,
    ExecutionEventHandler,
    ObservabilityLevel,
    Request,
    Response,
    Result,
    SafetyStatus,
)

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Sequence


class _Session:
    async def send_async(self, request: Request) -> Response:
        return Response(text=request.prompt or "")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        pass


class _Adapter:
    async def create_session_async(self) -> _Session:
        return _Session()

    @property
    def manifest(self) -> AppManifest:
        return AppManifest(name="trial-test")

    @property
    def observability_profile(self) -> ObservabilityLevel:
        return ObservabilityLevel.RESPONSE_ONLY


class _StaticExecution(BaseExecution):
    def __init__(
        self,
        *,
        result: Result,
        adapter_log: list[AgentAdapter] | None = None,
        event_log: list[str] | None = None,
        label: str = "",
        event_handlers: list[ExecutionEventHandler] | None = None,
    ) -> None:
        super().__init__(event_handlers=event_handlers)
        self._result = result
        self._adapter_log = adapter_log
        self._event_log = event_log
        self._label = label

    @property
    def strategy_name(self) -> str:
        return "static"

    async def _execute_async(self, *, adapter: AgentAdapter) -> Result:
        if self._adapter_log is not None:
            self._adapter_log.append(adapter)
        if self._event_log is not None:
            self._event_log.append(f"execute-{self._label}")
        return self._result


class _RaisingExecution(BaseExecution):
    def __init__(self, *, error: BaseException) -> None:
        super().__init__()
        self._error = error

    @property
    def strategy_name(self) -> str:
        return "raising"

    async def _execute_async(self, *, adapter: AgentAdapter) -> Result:
        raise self._error


class _MetadataObserver(ExecutionEventHandler):
    def __init__(self) -> None:
        self.post_metadata: list[dict[str, object]] = []

    async def on_event(self, *, event_data: ExecutionEventData) -> None:
        if (
            event_data.event is ExecutionEvent.ON_POST_EXECUTE
            and event_data.result is not None
        ):
            self.post_metadata.append(dict(event_data.result.metadata))


def _result(
    *,
    status: SafetyStatus = SafetyStatus.SAFE,
    summary: str = "result",
    metadata: dict[str, object] | None = None,
) -> Result:
    return Result(status=status, summary=summary, metadata=metadata or {})


def _factory_for(
    *,
    executions: Sequence[BaseExecution],
) -> Callable[[], BaseExecution]:
    iterator = iter(executions)
    return lambda: next(iterator)


def _unexpected_factory() -> BaseExecution:
    raise AssertionError("factory should not be called")


def _non_execution_factory() -> object:
    return object()


class TestTrialInputValidation:
    @pytest.mark.parametrize("count", [True, False, 1.0, "1", None])
    async def test_rejects_non_integer_count_async(self, count: object) -> None:
        with pytest.raises(TypeError, match="count must be a positive integer"):
            await execute_trials_async(
                execution_factory=_unexpected_factory,
                adapter=_Adapter(),
                count=count,  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize("count", [0, -1])
    async def test_rejects_non_positive_count_async(self, count: int) -> None:
        with pytest.raises(ValueError, match="greater than zero"):
            await execute_trials_async(
                execution_factory=_unexpected_factory,
                adapter=_Adapter(),
                count=count,
            )

    @pytest.mark.parametrize("threshold", [True, False, "0.5", None])
    async def test_rejects_non_numeric_threshold_async(
        self,
        threshold: object,
    ) -> None:
        with pytest.raises(TypeError, match="threshold must be a finite number"):
            await execute_trials_async(
                execution_factory=_unexpected_factory,
                adapter=_Adapter(),
                count=1,
                threshold=threshold,  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize(
        "threshold",
        [0.0, -0.1, 1.1, float("nan"), float("inf"), float("-inf"), 10**5000],
        ids=[
            "zero",
            "negative",
            "above-one",
            "nan",
            "positive-infinity",
            "negative-infinity",
            "overflowing-integer",
        ],
    )
    async def test_rejects_out_of_range_threshold_async(
        self,
        threshold: float,
    ) -> None:
        with pytest.raises(ValueError, match=r"finite number in \(0, 1\]"):
            await execute_trials_async(
                execution_factory=_unexpected_factory,
                adapter=_Adapter(),
                count=1,
                threshold=threshold,
            )

    async def test_accepts_integer_one_threshold_async(self) -> None:
        result = _result()
        batch = await execute_trials_async(
            execution_factory=lambda: _StaticExecution(result=result),
            adapter=_Adapter(),
            count=1,
            threshold=1,
        )
        assert batch.threshold == pytest.approx(1.0)


class TestTrialExecution:
    async def test_runs_exactly_n_times_sequentially_async(self) -> None:
        events: list[str] = []
        results = [_result(summary=str(index)) for index in range(3)]
        next_index = 0

        def factory() -> BaseExecution:
            nonlocal next_index
            index = next_index
            next_index += 1
            events.append(f"factory-{index}")
            return _StaticExecution(
                result=results[index],
                event_log=events,
                label=str(index),
            )

        batch = await execute_trials_async(
            execution_factory=factory,
            adapter=_Adapter(),
            count=3,
        )

        assert events == [
            "factory-0",
            "execute-0",
            "factory-1",
            "execute-1",
            "factory-2",
            "execute-2",
        ]
        assert batch.results == tuple(results)

    async def test_passes_same_adapter_identity_async(self) -> None:
        adapter = _Adapter()
        adapter_log: list[AgentAdapter] = []
        executions = [
            _StaticExecution(result=_result(), adapter_log=adapter_log)
            for _ in range(3)
        ]

        await execute_trials_async(
            execution_factory=_factory_for(executions=executions),
            adapter=adapter,
            count=3,
        )

        assert adapter_log == [adapter, adapter, adapter]
        assert all(seen is adapter for seen in adapter_log)

    async def test_rejects_non_execution_factory_product_async(self) -> None:
        with pytest.raises(TypeError, match="must return a BaseExecution"):
            await execute_trials_async(
                execution_factory=_non_execution_factory,  # ty: ignore[invalid-argument-type]
                adapter=_Adapter(),
                count=1,
            )

    async def test_rejects_reused_execution_identity_async(self) -> None:
        result = _result()
        execution = _StaticExecution(result=result)

        with pytest.raises(ValueError, match="distinct BaseExecution"):
            await execute_trials_async(
                execution_factory=lambda: execution,
                adapter=_Adapter(),
                count=2,
            )

        assert result.metadata[TRIAL_BATCH_INDEX_KEY] == 0

    async def test_rejects_reused_result_identity_async(self) -> None:
        result = _result()
        executions = [
            _StaticExecution(result=result),
            _StaticExecution(result=result),
        ]

        with pytest.raises(ValueError, match="distinct Result"):
            await execute_trials_async(
                execution_factory=_factory_for(executions=executions),
                adapter=_Adapter(),
                count=2,
            )

        assert result.metadata[TRIAL_BATCH_INDEX_KEY] == 0

    async def test_execution_error_becomes_result_and_batch_continues_async(
        self,
    ) -> None:
        executions = [
            _RaisingExecution(error=RuntimeError("failed")),
            _StaticExecution(result=_result()),
        ]

        batch = await execute_trials_async(
            execution_factory=_factory_for(executions=executions),
            adapter=_Adapter(),
            count=2,
            threshold=0.5,
        )

        assert [result.status for result in batch.results] == [
            SafetyStatus.ERROR,
            SafetyStatus.SAFE,
        ]
        assert batch.error_count == 1
        assert batch.passed is True


class TestTrialFailures:
    async def test_factory_error_propagates_after_partial_batch_async(self) -> None:
        first = _result()
        calls = 0
        error = RuntimeError("factory failed")

        def factory() -> BaseExecution:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise error
            return _StaticExecution(result=first)

        with pytest.raises(RuntimeError) as exc_info:
            await execute_trials_async(
                execution_factory=factory,
                adapter=_Adapter(),
                count=3,
            )

        assert exc_info.value is error
        assert first.metadata[TRIAL_BATCH_INDEX_KEY] == 0
        assert first.metadata[TRIAL_BATCH_COUNT_KEY] == 3

    async def test_factory_error_before_first_result_adds_no_metadata_async(
        self,
    ) -> None:
        error = RuntimeError("factory failed")

        def factory() -> BaseExecution:
            raise error

        with pytest.raises(RuntimeError) as exc_info:
            await execute_trials_async(
                execution_factory=factory,
                adapter=_Adapter(),
                count=2,
            )

        assert exc_info.value is error

    async def test_cancellation_propagates_after_partial_batch_async(self) -> None:
        first = _result()
        executions = [
            _StaticExecution(result=first),
            _RaisingExecution(error=asyncio.CancelledError()),
        ]

        with pytest.raises(asyncio.CancelledError):
            await execute_trials_async(
                execution_factory=_factory_for(executions=executions),
                adapter=_Adapter(),
                count=2,
            )

        assert first.metadata[TRIAL_BATCH_INDEX_KEY] == 0
        assert first.metadata[TRIAL_BATCH_COUNT_KEY] == 2


class TestTrialBatch:
    def test_public_exports(self) -> None:
        import rampart
        import rampart.core

        assert rampart.TrialBatch is TrialBatch
        assert rampart.core.TrialBatch is TrialBatch
        assert rampart.execute_trials_async is execute_trials_async
        assert rampart.core.execute_trials_async is execute_trials_async

    async def test_computes_all_status_counts_async(self) -> None:
        statuses = list(SafetyStatus)
        executions = [
            _StaticExecution(result=_result(status=status)) for status in statuses
        ]

        batch = await execute_trials_async(
            execution_factory=_factory_for(executions=executions),
            adapter=_Adapter(),
            count=4,
            threshold=0.25,
        )

        assert batch.safe_count == 1
        assert batch.unsafe_count == 1
        assert batch.undetermined_count == 1
        assert batch.error_count == 1
        assert batch.pass_rate == pytest.approx(0.25)
        assert batch.passed is True
        assert bool(batch) is True

    async def test_threshold_alone_allows_explicitly_tolerated_unsafe_async(
        self,
    ) -> None:
        statuses = [
            SafetyStatus.SAFE,
            SafetyStatus.SAFE,
            SafetyStatus.UNSAFE,
        ]
        executions = [
            _StaticExecution(result=_result(status=status)) for status in statuses
        ]

        batch = await execute_trials_async(
            execution_factory=_factory_for(executions=executions),
            adapter=_Adapter(),
            count=3,
            threshold=2 / 3,
        )

        assert batch.unsafe_count == 1
        assert batch.pass_rate == pytest.approx(2 / 3)
        assert batch.passed is True

    async def test_default_threshold_is_fail_closed_async(self) -> None:
        executions = [
            _StaticExecution(result=_result(status=SafetyStatus.SAFE)),
            _StaticExecution(result=_result(status=SafetyStatus.UNSAFE)),
        ]

        batch = await execute_trials_async(
            execution_factory=_factory_for(executions=executions),
            adapter=_Adapter(),
            count=2,
        )

        assert batch.threshold == pytest.approx(1.0)
        assert batch.passed is False
        assert bool(batch) is False

    async def test_preserves_original_results_and_freezes_outer_batch_async(
        self,
    ) -> None:
        result = _result()
        batch = await execute_trials_async(
            execution_factory=lambda: _StaticExecution(result=result),
            adapter=_Adapter(),
            count=1,
        )

        assert batch.results[0] is result
        attribute = "threshold"
        with pytest.raises(FrozenInstanceError):
            setattr(batch, attribute, 0.5)

    async def test_repr_contains_aggregates_without_evidence_async(self) -> None:
        evidence = "raw secret evidence"
        batch = await execute_trials_async(
            execution_factory=lambda: _StaticExecution(
                result=_result(summary=evidence, metadata={"secret": evidence}),
            ),
            adapter=_Adapter(),
            count=1,
        )

        rendered = repr(batch)
        assert "safe_count=1" in rendered
        assert "passed=True" in rendered
        assert evidence not in rendered
        assert "results=" not in rendered

    async def test_generates_canonical_uuid4_and_exact_metadata_async(self) -> None:
        results = [_result(), _result()]
        batch = await execute_trials_async(
            execution_factory=_factory_for(
                executions=[_StaticExecution(result=result) for result in results],
            ),
            adapter=_Adapter(),
            count=2,
            threshold=0.5,
        )

        parsed = UUID(batch.batch_id)
        assert parsed.version == 4
        assert str(parsed) == batch.batch_id
        assert [result.metadata[TRIAL_BATCH_INDEX_KEY] for result in results] == [0, 1]
        for result in results:
            assert result.metadata[TRIAL_BATCH_SCHEMA_KEY] == TRIAL_BATCH_SCHEMA
            assert result.metadata[TRIAL_BATCH_ID_KEY] == batch.batch_id
            assert result.metadata[TRIAL_BATCH_COUNT_KEY] == 2
            assert result.metadata[TRIAL_BATCH_THRESHOLD_KEY] == pytest.approx(0.5)

    async def test_metadata_merge_is_authoritative_and_preserves_user_keys_async(
        self,
    ) -> None:
        result = _result(
            metadata={
                "user": "kept",
                TRIAL_BATCH_SCHEMA_KEY: "wrong",
                TRIAL_BATCH_INDEX_KEY: 99,
            },
        )

        await execute_trials_async(
            execution_factory=lambda: _StaticExecution(result=result),
            adapter=_Adapter(),
            count=1,
        )

        assert result.metadata["user"] == "kept"
        assert result.metadata[TRIAL_BATCH_SCHEMA_KEY] == TRIAL_BATCH_SCHEMA
        assert result.metadata[TRIAL_BATCH_INDEX_KEY] == 0

    async def test_post_handler_runs_before_reporting_metadata_is_added_async(
        self,
    ) -> None:
        observer = _MetadataObserver()
        result = _result(metadata={"user": "visible"})

        await execute_trials_async(
            execution_factory=lambda: _StaticExecution(
                result=result,
                event_handlers=[observer],
            ),
            adapter=_Adapter(),
            count=1,
        )

        assert observer.post_metadata == [{"user": "visible"}]
        assert result.metadata[TRIAL_BATCH_SCHEMA_KEY] == TRIAL_BATCH_SCHEMA

    def test_constructor_rejects_noncanonical_id(self) -> None:
        with pytest.raises(ValueError, match="canonical UUID4"):
            TrialBatch(
                batch_id="not-a-uuid",
                results=(_result(),),
                requested_count=1,
            )
