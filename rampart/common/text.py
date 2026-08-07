# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Terminal presentation text helpers shared across RAMPART."""

from __future__ import annotations


def escape_terminal_controls(text: str) -> str:
    r"""Render terminal control characters as visible lowercase hex escapes.

    Args:
        text (str): The text to prepare for one-line terminal presentation.

    Returns:
        str: Text with C0, DEL, and C1 controls rendered as literal
            ``\xNN`` sequences. Printable Unicode is preserved.
    """
    c0_end = 0x20
    del_start = 0x7F
    c1_end = 0x9F
    escaped: list[str] = []
    for character in text:
        codepoint = ord(character)
        if codepoint < c0_end or del_start <= codepoint <= c1_end:
            escaped.append(f"\\x{codepoint:02x}")
        else:
            escaped.append(character)
    return "".join(escaped)
