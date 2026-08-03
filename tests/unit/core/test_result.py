# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for rampart.core.result.

Result, SafetyStatus, HarmCategory, resolve functions.
"""

import warnings

import pytest

from rampart.core.result import (
    HarmCategory,
    InjectionRecord,
    Result,
    SafetyStatus,
    resolve_as_attack,
    resolve_as_probe,
    resolve_attack_verdict,
    resolve_probe_verdict,
)
from rampart.core.types import (
    EvalOutcome,
    EvalResult,
    ObservabilityLevel,
    Request,
    Response,
    TerminationReason,
    Turn,
)


def _er(outcome: EvalOutcome) -> EvalResult:
    """Shorthand to build an EvalResult with a given outcome."""
    return EvalResult(outcome=outcome)


class TestSafetyStatus:
    def test_values(self) -> None:
        assert SafetyStatus.SAFE.value == "safe"
        assert SafetyStatus.UNSAFE.value == "unsafe"
        assert SafetyStatus.UNDETERMINED.value == "undetermined"
        assert SafetyStatus.ERROR.value == "error"


class TestHarmCategory:
    def test_is_strenum(self) -> None:
        assert isinstance(HarmCategory.PROMPT_INJECTION, str)

    def test_values_are_plain_strings(self) -> None:
        assert HarmCategory.PROMPT_INJECTION == "prompt_injection"
        assert HarmCategory.JAILBREAK == "jailbreak"
        assert HarmCategory.DATA_EXFILTRATION == "data_exfiltration"
        assert HarmCategory.OVER_PERMISSIVE_ACTION == "over_permissive_action"
        assert HarmCategory.DATA_LEAKAGE == "data_leakage"
        assert HarmCategory.CONTENT_SAFETY == "content_safety"
        assert HarmCategory.HALLUCINATION == "hallucination"
        assert HarmCategory.BEHAVIORAL_REGRESSION == "behavioral_regression"

    def test_xpia_is_not_a_harm_category(self) -> None:
        assert not hasattr(HarmCategory, "XPIA")

    def test_interchangeable_with_plain_string(self) -> None:
        assert HarmCategory.PROMPT_INJECTION == "prompt_injection"
        assert HarmCategory.PROMPT_INJECTION == "prompt_injection"

    def test_usable_as_dict_key(self) -> None:
        d: dict[str, int] = {HarmCategory.DATA_EXFILTRATION: 1, "custom_risk": 2}
        assert d["data_exfiltration"] == 1
        assert d[HarmCategory.DATA_EXFILTRATION] == 1


class TestInjectionRecord:
    def test_construction(self) -> None:
        rec = InjectionRecord(payload_id="abc123", surface_name="SharePoint")
        assert rec.payload_id == "abc123"
        assert rec.surface_name == "SharePoint"

    def test_none_payload_id(self) -> None:
        rec = InjectionRecord(payload_id=None, surface_name="Exchange")
        assert rec.payload_id is None


class TestResult:
    def test_bool_returns_safe_true(self) -> None:
        r = Result(status=SafetyStatus.SAFE, summary="ok")
        assert bool(r) is True

    def test_bool_returns_safe_false(self) -> None:
        r = Result(status=SafetyStatus.UNSAFE, summary="bad")
        assert bool(r) is False

    def test_assert_safe_pattern(self) -> None:
        safe_result = Result(status=SafetyStatus.SAFE, summary="ok")
        assert safe_result, safe_result.summary

        unsafe_result = Result(
            status=SafetyStatus.UNSAFE,
            summary="attack detected",
        )
        with pytest.raises(AssertionError):
            assert unsafe_result, unsafe_result.summary

    def test_repr(self) -> None:
        r = Result(status=SafetyStatus.SAFE, summary="Agent defended")
        assert "safe=True" in repr(r)
        assert "safe" in repr(r)
        assert "Agent defended" in repr(r)

    def test_defaults(self) -> None:
        r = Result(status=SafetyStatus.SAFE, summary="ok")
        assert r.turns == []
        assert r.eval_results == []
        assert r.duration_seconds == pytest.approx(0.0)
        assert r.harm_category is None
        assert r.strategy == ""
        assert r.observability_level is ObservabilityLevel.RESPONSE_ONLY
        assert r.injections == []
        assert r.metadata == {}
        assert r.evaluation is None
        assert r.termination_reason is None

    def test_final_evaluation_and_termination_reason_round_trip(self) -> None:
        evaluation = _er(EvalOutcome.DETECTED)
        r = Result(
            status=SafetyStatus.UNSAFE,
            summary="bad",
            evaluation=evaluation,
            termination_reason=TerminationReason.STOP_CONDITION,
        )
        assert r.evaluation is evaluation
        assert r.termination_reason is TerminationReason.STOP_CONDITION

    def test_harm_category_accepts_enum(self) -> None:
        r = Result(
            status=SafetyStatus.SAFE,
            summary="ok",
            harm_category=HarmCategory.DATA_EXFILTRATION,
        )
        assert r.harm_category == "data_exfiltration"
        assert r.harm_category == HarmCategory.DATA_EXFILTRATION

    def test_harm_category_accepts_plain_string(self) -> None:
        r = Result(
            status=SafetyStatus.SAFE,
            summary="ok",
            harm_category="custom_product_risk",
        )
        assert r.harm_category == "custom_product_risk"


class TestResultEvalResultsProperty:
    """eval_results is a property derived from turns."""

    def test_empty_turns_gives_empty_eval_results(self) -> None:
        r = Result(status=SafetyStatus.SAFE, summary="ok")
        assert r.eval_results == []

    def test_turns_with_eval_results_returned_in_order(self) -> None:
        er1 = _er(EvalOutcome.NOT_DETECTED)
        er2 = _er(EvalOutcome.DETECTED)
        turns = [
            Turn(
                request=Request(prompt="p1"),
                response=Response(text="r1"),
                eval_result=er1,
            ),
            Turn(
                request=Request(prompt="p2"),
                response=Response(text="r2"),
                eval_result=er2,
            ),
        ]
        r = Result(
            status=SafetyStatus.UNSAFE,
            summary="bad",
            turns=turns,
        )
        assert r.eval_results == [er1, er2]

    def test_turns_without_eval_result_filtered(self) -> None:
        er = _er(EvalOutcome.DETECTED)
        turns = [
            Turn(
                request=Request(prompt="p1"),
                response=Response(text="r1"),
            ),
            Turn(
                request=Request(prompt="p2"),
                response=Response(text="r2"),
                eval_result=er,
            ),
        ]
        r = Result(
            status=SafetyStatus.UNSAFE,
            summary="bad",
            turns=turns,
        )
        assert r.eval_results == [er]

    def test_final_evaluation_is_not_in_turn_eval_results(self) -> None:
        final = _er(EvalOutcome.DETECTED)
        turn_evaluation = _er(EvalOutcome.NOT_DETECTED)
        r = Result(
            status=SafetyStatus.UNSAFE,
            summary="bad",
            evaluation=final,
            turns=[
                Turn(
                    request=Request(prompt="p"),
                    response=Response(text="r"),
                    eval_result=turn_evaluation,
                ),
            ],
        )
        assert r.eval_results == [turn_evaluation]


class TestResolveAsAttack:
    def test_empty_returns_error(self) -> None:
        status = resolve_as_attack(eval_results=[])
        assert status is SafetyStatus.ERROR

    def test_detected_returns_unsafe(self) -> None:
        status = resolve_as_attack(
            eval_results=[_er(EvalOutcome.DETECTED)],
        )
        assert status is SafetyStatus.UNSAFE

    def test_not_detected_returns_safe(self) -> None:
        status = resolve_as_attack(
            eval_results=[_er(EvalOutcome.NOT_DETECTED)],
        )
        assert status is SafetyStatus.SAFE

    def test_undetermined_only_returns_undetermined(self) -> None:
        status = resolve_as_attack(
            eval_results=[_er(EvalOutcome.UNDETERMINED)],
        )
        assert status is SafetyStatus.UNDETERMINED

    def test_mixed_detected_and_undetermined_returns_unsafe(self) -> None:
        status = resolve_as_attack(
            eval_results=[
                _er(EvalOutcome.UNDETERMINED),
                _er(EvalOutcome.DETECTED),
            ],
        )
        assert status is SafetyStatus.UNSAFE

    def test_mixed_not_detected_and_undetermined_returns_undetermined(self) -> None:
        status = resolve_as_attack(
            eval_results=[
                _er(EvalOutcome.NOT_DETECTED),
                _er(EvalOutcome.UNDETERMINED),
            ],
        )
        assert status is SafetyStatus.UNDETERMINED

    def test_all_not_detected_returns_safe(self) -> None:
        status = resolve_as_attack(
            eval_results=[
                _er(EvalOutcome.NOT_DETECTED),
                _er(EvalOutcome.NOT_DETECTED),
            ],
        )
        assert status is SafetyStatus.SAFE


class TestResolveAsProbe:
    def test_empty_returns_error(self) -> None:
        status = resolve_as_probe(eval_results=[])
        assert status is SafetyStatus.ERROR

    def test_detected_returns_safe(self) -> None:
        status = resolve_as_probe(
            eval_results=[_er(EvalOutcome.DETECTED)],
        )
        assert status is SafetyStatus.SAFE

    def test_not_detected_returns_unsafe(self) -> None:
        status = resolve_as_probe(
            eval_results=[_er(EvalOutcome.NOT_DETECTED)],
        )
        assert status is SafetyStatus.UNSAFE

    def test_undetermined_only_returns_undetermined(self) -> None:
        status = resolve_as_probe(
            eval_results=[_er(EvalOutcome.UNDETERMINED)],
        )
        assert status is SafetyStatus.UNDETERMINED

    def test_mixed_not_detected_and_undetermined_returns_unsafe(self) -> None:
        status = resolve_as_probe(
            eval_results=[
                _er(EvalOutcome.UNDETERMINED),
                _er(EvalOutcome.NOT_DETECTED),
            ],
        )
        assert status is SafetyStatus.UNSAFE

    def test_mixed_detected_and_undetermined_returns_undetermined(self) -> None:
        status = resolve_as_probe(
            eval_results=[
                _er(EvalOutcome.DETECTED),
                _er(EvalOutcome.UNDETERMINED),
            ],
        )
        assert status is SafetyStatus.UNDETERMINED

    def test_all_detected_returns_safe(self) -> None:
        status = resolve_as_probe(
            eval_results=[
                _er(EvalOutcome.DETECTED),
                _er(EvalOutcome.DETECTED),
            ],
        )
        assert status is SafetyStatus.SAFE


def test_legacy_resolvers_remain_warning_free() -> None:
    """The additive API does not start the legacy deprecation clock."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_as_attack(eval_results=[]) is SafetyStatus.ERROR
        assert resolve_as_probe(eval_results=[]) is SafetyStatus.ERROR


class TestResolveAttackVerdict:
    @pytest.mark.parametrize(
        ("evaluation", "expected"),
        [
            (None, SafetyStatus.ERROR),
            (_er(EvalOutcome.DETECTED), SafetyStatus.UNSAFE),
            (_er(EvalOutcome.NOT_DETECTED), SafetyStatus.SAFE),
            (_er(EvalOutcome.UNDETERMINED), SafetyStatus.UNDETERMINED),
        ],
    )
    def test_maps_single_evaluation(
        self,
        evaluation: EvalResult | None,
        expected: SafetyStatus,
    ) -> None:
        assert resolve_attack_verdict(evaluation=evaluation) is expected


class TestResolveProbeVerdict:
    @pytest.mark.parametrize(
        ("evaluation", "expected"),
        [
            (None, SafetyStatus.ERROR),
            (_er(EvalOutcome.DETECTED), SafetyStatus.SAFE),
            (_er(EvalOutcome.NOT_DETECTED), SafetyStatus.UNSAFE),
            (_er(EvalOutcome.UNDETERMINED), SafetyStatus.UNDETERMINED),
        ],
    )
    def test_maps_single_evaluation(
        self,
        evaluation: EvalResult | None,
        expected: SafetyStatus,
    ) -> None:
        assert resolve_probe_verdict(evaluation=evaluation) is expected
