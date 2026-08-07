# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Terminal presentation text helpers shared across RAMPART."""

from __future__ import annotations

import traceback


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


def escape_terminal_repr(value: object) -> str:
    """Return an object's representation with terminal controls escaped.

    Args:
        value (object): The value to represent for terminal-visible logging.

    Returns:
        str: The value's ``repr`` with raw C0, DEL, and C1 controls escaped.
    """
    try:
        representation = repr(value)
    except Exception as exc:  # ruff: ignore[blind-except] — logging fallback
        value_type = escape_terminal_controls(type(value).__name__)
        error_type = escape_terminal_controls(type(exc).__name__)
        return f"<unrepresentable {value_type}; {error_type} raised by repr()>"
    return escape_terminal_controls(representation)


def format_exception_for_terminal(exception: BaseException) -> str:
    """Format a complete traceback for safe one-line terminal logging.

    Args:
        exception (BaseException): The caught exception to format.

    Returns:
        str: The chained traceback with terminal controls, including line
            breaks, rendered as visible lowercase hex escapes. If traceback
            formatting fails, returns a terminal-safe type-only diagnostic.
    """
    try:
        formatted = "".join(traceback.format_exception(exception)).removesuffix("\n")
    except Exception as exc:  # ruff: ignore[blind-except] — logging fallback
        exception_type = escape_terminal_controls(type(exception).__name__)
        error_type = escape_terminal_controls(type(exc).__name__)
        return (
            f"<traceback unavailable for {exception_type}; "
            f"{error_type} raised while formatting>"
        )
    return escape_terminal_controls(formatted)
