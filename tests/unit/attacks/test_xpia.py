# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rampart.attacks import Attacks
from rampart.core.errors import InfrastructureError
from rampart.core.evaluator import Evaluator
from rampart.core.injection import InjectionHandle
from rampart.core.manifest import AppManifest
from rampart.core.result import SafetyStatus
from rampart.core.types import (
    EvalContext,
    EvalOutcome,
    EvalResult,
    ObservabilityLevel,
    Payload,
    Request,
    Response,
    TerminationReason,
    ToolCall,
)
from rampart.evaluators import ResponseContains, ResponseScope, ToolCalled
from tests.fixtures import MockAdapter

_DEFAULT_MANIFEST = AppManifest(name="TestAgent")


def _mock_handle(
    *,
    surface_name: str = "FakeSurface",
    payload_id: str | None = "p-001",
) -> AsyncMock:
    """Create an AsyncMock satisfying the InjectionHandle protocol."""
    h = AsyncMock(spec=InjectionHandle)
    h.surface_name = surface_name
    h.payload_id = payload_id
    h.__aenter__.return_value = h
    return h


def _mock_evaluator(
    outcome: EvalOutcome,
    *,
    confidence: float = 1.0,
    evidence: list[str] | None = None,
    rationale: str = "",
) -> AsyncMock:
    """Create an AsyncMock evaluator returning a fixed EvalResult."""
    evaluator = AsyncMock(spec=Evaluator)
    evaluator.evaluate_async.return_value = EvalResult(
        outcome=outcome,
        confidence=confidence,
        evidence=evidence or [],
        rationale=rationale,
    )
    return evaluator


def _adapter(
    *,
    responses: list[Response] | None = None,
    observability: ObservabilityLevel = ObservabilityLevel.TOOL_AND_SIDE_EFFECTS,
) -> MockAdapter:
    """Shorthand for MockAdapter with sensible defaults."""
    return MockAdapter(
        responses=responses or [Response(text="ok")],
        manifest=_DEFAULT_MANIFEST,
        observability_profile=observability,
    )


class TestXPIADetection:
    """Attack semantics: DETECTED->UNSAFE, NOT_DETECTED->SAFE."""

    async def test_detected_returns_unsafe_with_evidence_in_summary(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(
                EvalOutcome.DETECTED,
                evidence=["exfil_call_found"],
            ),
        ).execute_async(adapter=_adapter())

        assert result.safe is False
        assert result.status is SafetyStatus.UNSAFE
        assert "exfil_call_found" in result.summary

    async def test_not_detected_returns_safe(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=_adapter())

        assert result.safe is True
        assert result.status is SafetyStatus.SAFE

    async def test_undetermined_returns_undetermined(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(
                EvalOutcome.UNDETERMINED,
                rationale="Insufficient signal",
            ),
        ).execute_async(adapter=_adapter())

        assert result.safe is False
        assert result.status is SafetyStatus.UNDETERMINED


class TestXPIAParameterValidation:
    def test_invalid_stop_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="'auto'"):
            Attacks.xpia(
                inject=_mock_handle(),
                trigger="Summarize Q3",
                evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
                stop_when="always",  # ty: ignore[invalid-argument-type]
            )


class TestXPIAEarlyStop:
    """Explicit and automatic attack stopping preserve justified costs."""

    async def test_explicit_identical_stop_reuses_fired_evaluation(self) -> None:
        evaluator = AsyncMock()
        evaluator.evaluate_async.side_effect = [
            EvalResult(outcome=EvalOutcome.NOT_DETECTED),
            EvalResult(outcome=EvalOutcome.DETECTED, evidence=["found_it"]),
        ]

        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["Turn 1", "Turn 2", "Turn 3"],
            evaluator=evaluator,
            stop_when=evaluator,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.UNSAFE
        assert len(result.turns) == 2
        assert result.termination_reason is TerminationReason.STOP_CONDITION
        assert evaluator.evaluate_async.await_count == 2

    async def test_explicit_non_firing_stop_reuses_terminal_evaluation(self) -> None:
        evaluator = _mock_evaluator(EvalOutcome.NOT_DETECTED)
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["Turn 1", "Turn 2", "Turn 3"],
            evaluator=evaluator,
            stop_when=evaluator,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.SAFE
        assert len(result.turns) == 3
        assert evaluator.evaluate_async.await_count == 3

    async def test_auto_stop_uses_absorbing_tool_evaluator(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["Turn 1", "Turn 2", "Turn 3"],
            evaluator=ToolCalled("exfiltrate"),
        ).execute_async(
            adapter=_adapter(
                responses=[
                    Response(text="one"),
                    Response(
                        text="two",
                        tool_calls=[ToolCall(name="exfiltrate")],
                    ),
                ],
            ),
        )

        assert result.status is SafetyStatus.UNSAFE
        assert len(result.turns) == 2
        assert result.termination_reason is TerminationReason.STOP_CONDITION

    async def test_auto_does_not_stop_unknown_evaluator(self) -> None:
        evaluator = _mock_evaluator(EvalOutcome.DETECTED)
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["Turn 1", "Turn 2", "Turn 3"],
            evaluator=evaluator,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.UNSAFE
        assert len(result.turns) == 3
        assert evaluator.evaluate_async.await_count == 1

    async def test_none_disables_auto_stop(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["Turn 1", "Turn 2"],
            evaluator=ToolCalled("exfiltrate"),
            stop_when=None,
        ).execute_async(
            adapter=_adapter(
                responses=[
                    Response(tool_calls=[ToolCall(name="exfiltrate")], text="one"),
                    Response(text="two"),
                ],
            ),
        )

        assert result.status is SafetyStatus.UNSAFE
        assert len(result.turns) == 2
        assert result.termination_reason is TerminationReason.DRIVER_EXHAUSTED
        assert result.evaluation is not None

    async def test_any_turn_response_evaluator_is_auto_stoppable(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["Turn 1", "Turn 2"],
            evaluator=ResponseContains("secret", scope=ResponseScope.ANY_TURN),
        ).execute_async(
            adapter=_adapter(
                responses=[Response(text="secret"), Response(text="clean")],
            ),
        )

        assert len(result.turns) == 1
        assert result.status is SafetyStatus.UNSAFE


class TestXPIAMaxTurns:
    """Max turns resolves normally from the terminal evaluation."""

    async def test_max_turns_resolves_normally(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["p1", "p2", "p3"],
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
            max_turns=2,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.SAFE
        assert len(result.turns) == 2
        assert result.termination_reason is TerminationReason.MAX_TURNS

    async def test_zero_turn_budget_returns_error(self) -> None:
        evaluator = AsyncMock(spec=Evaluator)
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="unused",
            evaluator=evaluator,
            max_turns=0,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.ERROR
        assert result.termination_reason is TerminationReason.MAX_TURNS
        evaluator.evaluate_async.assert_not_awaited()


class TestXPIACleanup:
    """Injection handles are always activated and cleaned up."""

    async def test_handle_entered_and_exited(self) -> None:
        handle = _mock_handle()

        await Attacks.xpia(
            inject=handle,
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=_adapter())

        handle.__aenter__.assert_awaited_once()
        handle.__aexit__.assert_awaited_once()
        handle.wait_until_ready.assert_awaited_once()

    async def test_multiple_handles_all_cleaned(self) -> None:
        h1 = _mock_handle(surface_name="SP")
        h2 = _mock_handle(surface_name="Exchange")

        await Attacks.xpia(
            inject=[h1, h2],
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=_adapter())

        for h in (h1, h2):
            h.__aenter__.assert_awaited_once()
            h.__aexit__.assert_awaited_once()
            h.wait_until_ready.assert_awaited_once()

    async def test_cleanup_on_evaluator_exception(self) -> None:
        """Handles are cleaned up even if the evaluator raises."""
        handle = _mock_handle()
        evaluator = AsyncMock()
        evaluator.evaluate_async.side_effect = RuntimeError("evaluator boom")

        result = await Attacks.xpia(
            inject=handle,
            trigger="Summarize Q3",
            evaluator=evaluator,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.ERROR
        assert "evaluator boom" in result.summary
        handle.__aexit__.assert_awaited_once()

    async def test_terminal_evaluation_runs_before_injection_cleanup(self) -> None:
        handle = _mock_handle()

        def evaluate(*, context: EvalContext) -> EvalResult:
            assert context.turns
            handle.__aexit__.assert_not_awaited()
            return EvalResult(outcome=EvalOutcome.NOT_DETECTED)

        evaluator = AsyncMock(spec=Evaluator)
        evaluator.evaluate_async.side_effect = evaluate

        result = await Attacks.xpia(
            inject=handle,
            trigger="Summarize Q3",
            evaluator=evaluator,
            stop_when=None,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.SAFE
        handle.__aexit__.assert_awaited_once()

    async def test_cleanup_failure_discards_successful_evaluation(self) -> None:
        handle = _mock_handle()
        handle.__aexit__.side_effect = RuntimeError("cleanup failed")

        result = await Attacks.xpia(
            inject=handle,
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
            stop_when=None,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.ERROR
        assert result.evaluation is None
        assert result.termination_reason is None


class TestXPIAZeroTurns:
    async def test_empty_trigger_returns_error_without_evaluation(self) -> None:
        evaluator = AsyncMock(spec=Evaluator)

        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=[],
            evaluator=evaluator,
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.ERROR
        assert result.evaluation is None
        assert result.termination_reason is TerminationReason.DRIVER_EXHAUSTED
        evaluator.evaluate_async.assert_not_awaited()


class TestXPIAInfrastructureError:
    """InfrastructureError produces ERROR result (base class concern)."""

    async def test_handle_activation_failure(self) -> None:
        handle = _mock_handle()
        handle.__aenter__.side_effect = InfrastructureError("SharePoint 503")

        result = await Attacks.xpia(
            inject=handle,
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.DETECTED),
        ).execute_async(adapter=_adapter())

        assert result.status is SafetyStatus.ERROR
        assert "SharePoint 503" in result.summary

    async def test_session_creation_failure(self) -> None:
        adapter = AsyncMock()
        adapter.create_session_async.side_effect = InfrastructureError(
            "Connection refused",
        )
        adapter.manifest = _DEFAULT_MANIFEST
        adapter.observability_profile = ObservabilityLevel.TOOL_AND_SIDE_EFFECTS

        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.DETECTED),
        ).execute_async(adapter=adapter)

        assert result.status is SafetyStatus.ERROR
        assert "Connection refused" in result.summary


class TestXPIAObservabilityAdjustment:
    """SAFE is downgraded to UNDETERMINED when observability is insufficient."""

    async def test_response_only_no_tools_downgrades_to_undetermined(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(
            adapter=_adapter(observability=ObservabilityLevel.RESPONSE_ONLY),
        )

        assert result.safe is False
        assert result.status is SafetyStatus.UNDETERMINED
        assert result.metadata["observability_adjusted"] is True
        assert "Insufficient observability" in result.summary

    async def test_adjustment_does_not_mutate_response_metadata(self) -> None:
        response_metadata = {"conversation_id": "c-01"}
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
            stop_when=None,
        ).execute_async(
            adapter=_adapter(
                responses=[Response(text="ok", metadata=response_metadata)],
                observability=ObservabilityLevel.RESPONSE_ONLY,
            ),
        )

        assert result.metadata["observability_adjusted"] is True
        assert "observability_adjusted" not in response_metadata
        assert "observability_adjusted" not in result.turns[0].response.metadata
        assert result.metadata is not result.turns[0].response.metadata

    async def test_response_only_with_tool_calls_stays_safe(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(
            adapter=_adapter(
                responses=[Response(text="ok", tool_calls=[ToolCall(name="search")])],
                observability=ObservabilityLevel.RESPONSE_ONLY,
            ),
        )

        assert result.safe is True
        assert result.status is SafetyStatus.SAFE

    async def test_non_response_only_levels_are_not_downgraded(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(
            adapter=_adapter(observability=ObservabilityLevel.TOOL_ONLY),
        )

        assert result.safe is True
        assert result.status is SafetyStatus.SAFE


class TestXPIAInjectionRecords:
    """Result carries injection records for reproduction."""

    async def test_single_handle_recorded(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(surface_name="SharePoint", payload_id="px-42"),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=_adapter())

        assert len(result.injections) == 1
        assert result.injections[0].payload_id == "px-42"
        assert result.injections[0].surface_name == "SharePoint"

    async def test_multi_handle_records(self) -> None:
        result = await Attacks.xpia(
            inject=[
                _mock_handle(surface_name="SP", payload_id="p1"),
                _mock_handle(surface_name="Exchange", payload_id="p2"),
            ],
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=_adapter())

        assert len(result.injections) == 2
        names = {r.surface_name for r in result.injections}
        assert names == {"SP", "Exchange"}


class TestXPIAAttachments:
    """Inline attachments flow through to turns via Request."""

    async def test_attachments_recorded_in_turns(self) -> None:
        attachment = Payload(content="malicious doc", id="att-1")

        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=Request(prompt="Open the attached file", attachments=[attachment]),
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=_adapter())

        assert result.turns[0].request.attachments[0].id == "att-1"


class TestResponseMetadataPropagation:
    """Response.metadata from the adapter flows into Result.metadata."""

    async def test_single_turn_metadata_promoted_to_top_level(self) -> None:
        adapter = _adapter(
            responses=[Response(text="ok", metadata={"conversation_id": "c-01"})],
        )
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=adapter)

        assert result.metadata == {"conversation_id": "c-01"}

    async def test_empty_response_metadata_produces_empty_result_metadata(self) -> None:
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger="Summarize Q3",
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=_adapter())

        assert result.metadata == {}

    async def test_multi_turn_metadata_keyed_by_turn_number(self) -> None:
        turn_0_metadata = {"page_url": "url0"}
        turn_1_metadata = {"page_url": "url1"}
        adapter = _adapter(
            responses=[
                Response(text="turn0", metadata=turn_0_metadata),
                Response(text="turn1", metadata=turn_1_metadata),
            ],
        )
        result = await Attacks.xpia(
            inject=_mock_handle(),
            trigger=["Summarize Q3", "Tell me more"],
            evaluator=_mock_evaluator(EvalOutcome.NOT_DETECTED),
        ).execute_async(adapter=adapter)

        assert "turn_0" in result.metadata
        assert result.metadata["turn_0"]["page_url"] == "url0"
        assert result.metadata["turn_1"]["page_url"] == "url1"
        result.metadata["turn_0"]["page_url"] = "changed"
        assert turn_0_metadata["page_url"] == "url0"
        assert turn_1_metadata["page_url"] == "url1"
