# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for SingleTurnExecution and Probes namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from rampart.core.errors import InfrastructureError
from rampart.core.evaluator import BaseEvaluator
from rampart.core.manifest import AppManifest
from rampart.core.prompt_driver import PromptDecision
from rampart.core.result import SafetyStatus
from rampart.core.types import (
    EvalContext,
    EvalOutcome,
    EvalResult,
    ObservabilityLevel,
    Request,
    Response,
    TerminationReason,
    ToolCall,
    Turn,
)
from rampart.drivers.static import StaticDriver
from rampart.evaluators import ResponseContains, ResponseScope
from rampart.probes import Probes
from tests.fixtures import MockAdapter

if TYPE_CHECKING:
    import types
    from typing import Self


def _adapter(*, responses: list[Response]) -> MockAdapter:
    """Build a MockAdapter for testing."""
    return MockAdapter(
        responses=responses,
        manifest=AppManifest(name="test-agent"),
    )


class _DetectsAlways(BaseEvaluator):
    """Evaluator stub that always detects."""

    async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
        return EvalResult(outcome=EvalOutcome.DETECTED, rationale="always detected")


class _NeverDetects(BaseEvaluator):
    """Evaluator stub that never detects."""

    async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
        return EvalResult(outcome=EvalOutcome.NOT_DETECTED, rationale="never detected")


class _DetectsToolCall(BaseEvaluator):
    """Evaluator stub that detects when a specific tool is called."""

    def __init__(self, *, tool_name: str) -> None:
        self._tool_name = tool_name

    async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
        for tc in context.all_tool_calls:
            if tc.name == self._tool_name:
                return EvalResult(
                    outcome=EvalOutcome.DETECTED,
                    rationale=f"Found {self._tool_name}",
                )
        return EvalResult(
            outcome=EvalOutcome.NOT_DETECTED,
            rationale=f"{self._tool_name} not called",
        )


class _ToolSequence(BaseEvaluator):
    """Detect a tool sequence across the complete trace."""

    def __init__(self, *tool_names: str) -> None:
        self._tool_names = tool_names

    async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
        observed = [call.name for call in context.all_tool_calls]
        next_index = 0
        for name in observed:
            if (
                next_index < len(self._tool_names)
                and name == self._tool_names[next_index]
            ):
                next_index += 1
        return EvalResult(
            outcome=(
                EvalOutcome.DETECTED
                if next_index == len(self._tool_names)
                else EvalOutcome.NOT_DETECTED
            ),
            rationale=f"Observed tools: {observed}",
        )


class _RequiresPrior(BaseEvaluator):
    """Detect when an action is always preceded by a required tool."""

    def __init__(self, *, action: str, required: str) -> None:
        self._action = action
        self._required = required

    async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
        seen_required = False
        for call in context.all_tool_calls:
            if call.name == self._required:
                seen_required = True
            if call.name == self._action and not seen_required:
                return EvalResult(
                    outcome=EvalOutcome.NOT_DETECTED,
                    rationale=f"{self._action} occurred before {self._required}",
                )
        return EvalResult(
            outcome=EvalOutcome.DETECTED,
            rationale=f"{self._required} preceded {self._action}",
        )


class TestProbePolarity:
    """Probe polarity: DETECTED -> SAFE, NOT_DETECTED -> UNSAFE."""

    async def test_detected_means_safe_async(self) -> None:
        adapter = _adapter(responses=[Response(text="ok")])

        result = await Probes.behavior(
            prompt="test",
            evaluator=_DetectsAlways(),
        ).execute_async(adapter=adapter)

        assert result.safe is True
        assert result.status == SafetyStatus.SAFE

    async def test_not_detected_means_unsafe_async(self) -> None:
        adapter = _adapter(responses=[Response(text="nope")])

        result = await Probes.behavior(
            prompt="test",
            evaluator=_NeverDetects(),
        ).execute_async(adapter=adapter)

        assert result.safe is False
        assert result.status == SafetyStatus.UNSAFE


class TestProbeStrategyName:
    """strategy_name is 'probe'."""

    async def test_strategy_name_async(self) -> None:
        adapter = _adapter(responses=[Response(text="ok")])

        result = await Probes.behavior(
            prompt="test",
            evaluator=_DetectsAlways(),
        ).execute_async(adapter=adapter)

        assert result.strategy == "probe"


class TestProbePromptCoercion:
    """Probes.behavior accepts str, list[str], and PromptDriver."""

    async def test_str_prompt_async(self) -> None:
        adapter = _adapter(responses=[Response(text="ok")])

        result = await Probes.behavior(
            prompt="hello",
            evaluator=_DetectsAlways(),
        ).execute_async(adapter=adapter)

        assert result.safe is True
        assert len(result.turns) == 1
        assert result.turns[0].request.prompt == "hello"

    async def test_list_prompt_async(self) -> None:
        adapter = _adapter(responses=[Response(text="ok")])

        result = await Probes.behavior(
            prompts=["first", "second"],
            evaluator=_NeverDetects(),
        ).execute_async(adapter=adapter)

        assert len(result.turns) == 2
        assert result.turns[0].request.prompt == "first"
        assert result.turns[1].request.prompt == "second"

    async def test_prompt_driver_async(self) -> None:
        prompt_driver = StaticDriver(prompts=["driven"])
        adapter = _adapter(responses=[Response(text="ok")])

        result = await Probes.behavior(
            driver=prompt_driver,
            evaluator=_DetectsAlways(),
        ).execute_async(adapter=adapter)

        assert result.turns[0].request.prompt == "driven"


class TestProbeParameterValidation:
    """Validates mutual-exclusion of prompt, prompts, and driver parameters."""

    def test_both_prompt_and_driver_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Probes.behavior(  # ty: ignore[no-matching-overload]
                prompt="hello",
                driver=StaticDriver(prompts=["driven"]),
                evaluator=_DetectsAlways(),
            )

    def test_both_prompt_and_prompts_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Probes.behavior(  # ty: ignore[no-matching-overload]
                prompt="hello",
                prompts=["a", "b"],
                evaluator=_DetectsAlways(),
            )

    def test_no_source_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Probes.behavior(evaluator=_DetectsAlways())  # ty: ignore[no-matching-overload]


class TestProbeInfrastructureError:
    """InfrastructureError becomes an ERROR result."""

    async def test_infra_error_produces_error_result_async(self) -> None:
        class FailingAdapter:
            manifest = AppManifest(name="broken")
            observability_profile = ObservabilityLevel.RESPONSE_ONLY

            async def create_session_async(self):
                raise InfrastructureError("Service down")

        result = await Probes.behavior(
            prompt="test",
            evaluator=_DetectsAlways(),
        ).execute_async(adapter=FailingAdapter())

        assert result.safe is False
        assert result.status == SafetyStatus.ERROR
        assert "InfrastructureError" in result.summary
        assert result.evaluation is None
        assert result.termination_reason is None


class TestProbeEndToEnd:
    """Probe flow works end-to-end with MockAdapter."""

    async def test_tool_call_detected_async(self) -> None:
        adapter = _adapter(
            responses=[
                Response(
                    text="Confirming action.",
                    tool_calls=[ToolCall(name="confirm_action")],
                ),
            ],
        )

        result = await Probes.behavior(
            prompt="Delete all my calendar events",
            evaluator=_DetectsToolCall(tool_name="confirm_action"),
        ).execute_async(adapter=adapter)

        assert result.safe is True
        assert result.status == SafetyStatus.SAFE

    async def test_tool_call_not_detected_async(self) -> None:
        adapter = _adapter(
            responses=[
                Response(text="Done. All events deleted."),
            ],
        )

        result = await Probes.behavior(
            prompt="Delete all my calendar events",
            evaluator=_DetectsToolCall(tool_name="confirm_action"),
        ).execute_async(adapter=adapter)

        assert result.safe is False
        assert result.status == SafetyStatus.UNSAFE

    async def test_assert_pattern_async(self) -> None:
        adapter = _adapter(
            responses=[
                Response(
                    text="Confirming.",
                    tool_calls=[ToolCall(name="confirm_action")],
                ),
            ],
        )

        result = await Probes.behavior(
            prompt="Delete all events",
            evaluator=_DetectsToolCall(tool_name="confirm_action"),
        ).execute_async(adapter=adapter)

        assert result, result.summary


class TestProbeMaxTurns:
    """Max turns resolves normally from the terminal evaluation."""

    async def test_max_turns_resolves_normally_async(self) -> None:
        adapter = _adapter(responses=[Response(text="ok")])

        result = await Probes.behavior(
            prompts=["a", "b", "c"],
            evaluator=_NeverDetects(),
            max_turns=2,
        ).execute_async(adapter=adapter)

        assert result.safe is False
        assert result.status == SafetyStatus.UNSAFE
        assert len(result.turns) == 2
        assert result.termination_reason is TerminationReason.MAX_TURNS
        assert "turn budget exhausted" in result.summary


class TestProbeFinalTraceCadence:
    async def test_verdict_evaluator_runs_once_over_complete_trace(self) -> None:
        evaluator = AsyncMock()
        evaluator.evaluate_async.return_value = EvalResult(
            outcome=EvalOutcome.DETECTED,
        )
        adapter = _adapter(
            responses=[Response(text="r1"), Response(text="r2"), Response(text="r3")],
        )

        result = await Probes.behavior(
            prompts=["p1", "p2", "p3"],
            evaluator=evaluator,
        ).execute_async(adapter=adapter)

        evaluator.evaluate_async.assert_awaited_once()
        context = evaluator.evaluate_async.await_args.kwargs["context"]
        assert len(context.turns) == 3
        assert result.evaluation is evaluator.evaluate_async.return_value
        assert result.eval_results == []
        assert result.termination_reason is TerminationReason.DRIVER_EXHAUSTED

    async def test_tool_sequence_resolves_from_complete_trace(self) -> None:
        result = await Probes.behavior(
            prompts=["first", "second"],
            evaluator=_ToolSequence("a", "b"),
        ).execute_async(
            adapter=_adapter(
                responses=[
                    Response(text="one", tool_calls=[ToolCall(name="a")]),
                    Response(text="two", tool_calls=[ToolCall(name="b")]),
                ],
            ),
        )

        assert result.status is SafetyStatus.SAFE
        assert len(result.turns) == 2

    async def test_requires_prior_observes_action_before_resolving(self) -> None:
        result = await Probes.behavior(
            prompts=["confirm", "delete"],
            evaluator=_RequiresPrior(action="delete", required="confirm"),
        ).execute_async(
            adapter=_adapter(
                responses=[
                    Response(text="confirmed", tool_calls=[ToolCall(name="confirm")]),
                    Response(text="deleted", tool_calls=[ToolCall(name="delete")]),
                ],
            ),
        )

        assert result.status is SafetyStatus.SAFE
        assert len(result.turns) == 2

    async def test_zero_turns_returns_error_without_evaluation(self) -> None:
        evaluator = AsyncMock()
        result = await Probes.behavior(
            prompts=[],
            evaluator=evaluator,
        ).execute_async(adapter=_adapter(responses=[Response(text="unused")]))

        assert result.status is SafetyStatus.ERROR
        assert result.evaluation is None
        assert result.termination_reason is TerminationReason.DRIVER_EXHAUSTED
        evaluator.evaluate_async.assert_not_awaited()

    async def test_zero_turn_budget_returns_error_with_budget_reason(self) -> None:
        evaluator = AsyncMock()
        result = await Probes.behavior(
            prompts=["unused"],
            evaluator=evaluator,
            max_turns=0,
        ).execute_async(adapter=_adapter(responses=[Response(text="unused")]))

        assert result.status is SafetyStatus.ERROR
        assert result.termination_reason is TerminationReason.MAX_TURNS
        assert "budget" in result.summary.lower()
        evaluator.evaluate_async.assert_not_awaited()

    async def test_default_driver_history_has_no_evaluator_feedback(self) -> None:
        class RecordingDriver:
            def __init__(self) -> None:
                self.histories: list[list[Turn]] = []

            async def next_prompt_async(
                self,
                *,
                history: list[Turn],
            ) -> PromptDecision | None:
                self.histories.append(history)
                if len(history) >= 2:
                    return None
                return PromptDecision(request=Request(prompt=f"p{len(history)}"))

        driver = RecordingDriver()
        result = await Probes.behavior(
            driver=driver,
            evaluator=_DetectsAlways(),
        ).execute_async(
            adapter=_adapter(responses=[Response(text="r1"), Response(text="r2")]),
        )

        assert len(result.turns) == 2
        assert all(
            turn.eval_result is None for history in driver.histories for turn in history
        )

    async def test_explicit_identical_stop_reuses_fired_evaluation(self) -> None:
        evaluator = AsyncMock()
        evaluator.evaluate_async.return_value = EvalResult(
            outcome=EvalOutcome.DETECTED,
            rationale="stop now",
        )

        result = await Probes.behavior(
            prompts=["p1", "p2"],
            evaluator=evaluator,
            stop_when=evaluator,
        ).execute_async(adapter=_adapter(responses=[Response(text="r1")]))

        assert len(result.turns) == 1
        assert result.status is SafetyStatus.SAFE
        assert result.termination_reason is TerminationReason.STOP_CONDITION
        assert evaluator.evaluate_async.await_count == 1

    async def test_distinct_stop_and_verdict_evaluators_do_not_cross_reuse(
        self,
    ) -> None:
        stop = AsyncMock()
        stop.evaluate_async.side_effect = [
            EvalResult(outcome=EvalOutcome.NOT_DETECTED),
            EvalResult(outcome=EvalOutcome.DETECTED),
        ]
        verdict = AsyncMock()
        verdict.evaluate_async.return_value = EvalResult(
            outcome=EvalOutcome.DETECTED,
            rationale="terminal verdict",
        )

        result = await Probes.behavior(
            prompts=["p1", "p2", "p3"],
            evaluator=verdict,
            stop_when=stop,
        ).execute_async(
            adapter=_adapter(responses=[Response(text="r1"), Response(text="r2")]),
        )

        assert len(result.turns) == 2
        assert stop.evaluate_async.await_count == 2
        verdict.evaluate_async.assert_awaited_once()
        context = verdict.evaluate_async.await_args.kwargs["context"]
        assert len(context.turns) == 2
        assert result.evaluation is verdict.evaluate_async.return_value

    async def test_explicit_stop_feedback_is_available_to_driver(self) -> None:
        class RecordingDriver:
            def __init__(self) -> None:
                self.histories: list[list[Turn]] = []

            async def next_prompt_async(
                self,
                *,
                history: list[Turn],
            ) -> PromptDecision | None:
                self.histories.append(history)
                if len(history) >= 2:
                    return None
                return PromptDecision(request=Request(prompt=f"p{len(history)}"))

        stop = AsyncMock()
        stop.evaluate_async.side_effect = [
            EvalResult(outcome=EvalOutcome.NOT_DETECTED, rationale="continue"),
            EvalResult(outcome=EvalOutcome.DETECTED, rationale="stop"),
        ]
        driver = RecordingDriver()

        await Probes.behavior(
            driver=driver,
            evaluator=stop,
            stop_when=stop,
        ).execute_async(
            adapter=_adapter(responses=[Response(text="r1"), Response(text="r2")]),
        )

        second_history = driver.histories[1]
        assert second_history[0].eval_result is not None
        assert second_history[0].eval_result.rationale == "continue"

    async def test_all_turns_scope_applies_to_complete_probe_trace(self) -> None:
        result = await Probes.behavior(
            prompts=["p1", "p2"],
            evaluator=ResponseContains("ok", scope=ResponseScope.ALL_TURNS),
        ).execute_async(
            adapter=_adapter(responses=[Response(text="no"), Response(text="ok")]),
        )

        assert len(result.turns) == 2
        assert result.status is SafetyStatus.UNSAFE

    async def test_negated_any_turn_scope_applies_to_complete_probe_trace(
        self,
    ) -> None:
        result = await Probes.behavior(
            prompts=["p1", "p2"],
            evaluator=~ResponseContains(
                "forbidden",
                scope=ResponseScope.ANY_TURN,
            ),
        ).execute_async(
            adapter=_adapter(responses=[Response(text="clean"), Response(text="safe")]),
        )

        assert result.status is SafetyStatus.SAFE

    async def test_unspecified_scope_warns_through_probe_execution(self) -> None:
        with pytest.warns(FutureWarning, match="ResponseScope"):
            result = await Probes.behavior(
                prompts=["p1", "p2"],
                evaluator=ResponseContains("ok"),
            ).execute_async(
                adapter=_adapter(
                    responses=[Response(text="not yet"), Response(text="ok")],
                ),
            )

        assert result.status is SafetyStatus.SAFE

    async def test_terminal_evaluation_runs_before_session_close(self) -> None:
        class RecordingSession:
            def __init__(self) -> None:
                self.closed = False

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
                self.closed = True

        session = RecordingSession()

        class Adapter:
            manifest = AppManifest(name="test-agent")
            observability_profile = ObservabilityLevel.RESPONSE_ONLY

            async def create_session_async(self):
                return session

        class CheckingEvaluator(BaseEvaluator):
            async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
                assert session.closed is False
                return EvalResult(outcome=EvalOutcome.DETECTED)

        result = await Probes.behavior(
            prompt="hello",
            evaluator=CheckingEvaluator(),
        ).execute_async(adapter=Adapter())

        assert result.status is SafetyStatus.SAFE
        assert session.closed is True

    async def test_safe_summary_includes_terminal_evidence(self) -> None:
        evaluator = AsyncMock()
        evaluator.evaluate_async.return_value = EvalResult(
            outcome=EvalOutcome.DETECTED,
            evidence=["terminal evidence"],
            rationale="terminal rationale",
        )

        result = await Probes.behavior(
            prompt="p",
            evaluator=evaluator,
        ).execute_async(adapter=_adapter(responses=[Response(text="r")]))

        assert "terminal evidence" in result.summary

    async def test_undetermined_summary_includes_terminal_rationale(self) -> None:
        evaluator = AsyncMock()
        evaluator.evaluate_async.return_value = EvalResult(
            outcome=EvalOutcome.UNDETERMINED,
            rationale="not enough evidence",
        )

        result = await Probes.behavior(
            prompt="p",
            evaluator=evaluator,
        ).execute_async(adapter=_adapter(responses=[Response(text="r")]))

        assert result.status is SafetyStatus.UNDETERMINED
        assert "not enough evidence" in result.summary
