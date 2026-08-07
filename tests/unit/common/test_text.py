# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from rampart.common.text import escape_terminal_controls


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
