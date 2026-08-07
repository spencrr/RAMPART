# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from rampart.common.text import (
    escape_terminal_controls,
    escape_terminal_repr,
    format_exception_for_terminal,
)


class TestEscapeTerminalControls:
    @pytest.mark.parametrize(
        "codepoint",
        [*range(0x20), 0x7F, *range(0x80, 0xA0)],
    )
    def test_escapes_every_c0_del_and_c1_control(self, codepoint: int) -> None:
        assert escape_terminal_controls(chr(codepoint)) == f"\\x{codepoint:02x}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("\x1b[31mred\x1b[0m", r"\x1b[31mred\x1b[0m"),
            ("\x1b]0;title\x07", r"\x1b]0;title\x07"),
            ("\x9b31mred", r"\x9b31mred"),
            ("\x9dtitle\x9c", r"\x9dtitle\x9c"),
            ("a\tb\nc\rd", r"a\x09b\x0ac\x0dd"),
        ],
    )
    def test_escapes_sequence_constituents(self, text: str, expected: str) -> None:
        assert escape_terminal_controls(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "hello world",
            "not an escape [0m or [31m here",
            "雪だるま ☃",
            "\u202eformat policy deferred",
        ],
    )
    def test_preserves_printable_text(self, text: str) -> None:
        assert escape_terminal_controls(text) == text

    def test_is_idempotent(self) -> None:
        escaped = escape_terminal_controls("\x1b[31mred\n")
        assert escape_terminal_controls(escaped) == escaped


class TestEscapeTerminalRepr:
    def test_escapes_controls_from_custom_repr(self) -> None:
        class HostileRepr:
            def __repr__(self) -> str:
                return "repr\x1b\t\n\r\x7f\x9b雪"

        assert escape_terminal_repr(HostileRepr()) == (
            r"repr\x1b\x09\x0a\x0d\x7f\x9b雪"
        )

    def test_falls_back_when_custom_repr_raises(self) -> None:
        class BrokenRepr:
            def __repr__(self) -> str:
                raise RuntimeError("broken\x1b\nrepr")

        BrokenRepr.__name__ = "Broken\x1b\nType\x9b"

        assert escape_terminal_repr(BrokenRepr()) == (
            r"<unrepresentable Broken\x1b\x0aType\x9b; "
            "RuntimeError raised by repr()>"
        )


class TestFormatExceptionForTerminal:
    def test_preserves_traceback_with_controls_escaped(self) -> None:
        raw_message = "boom\x1b\t\n\r\x7f\x9b雪"
        try:
            raise RuntimeError(raw_message)
        except RuntimeError as exc:
            caught = exc
            formatted = format_exception_for_terminal(exc)

        assert "Traceback (most recent call last):" in formatted
        assert "test_preserves_traceback_with_controls_escaped" in formatted
        assert r"RuntimeError: boom\x1b\x09\x0a\x0d\x7f\x9b雪" in formatted
        assert "\x1b" not in formatted
        assert "\t" not in formatted
        assert "\n" not in formatted
        assert "\r" not in formatted
        assert "\x7f" not in formatted
        assert "\x9b" not in formatted
        assert str(caught) == raw_message
