# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded diagnostic rendering for pytest plugin validation boundaries."""

from __future__ import annotations

from itertools import islice
from typing import Any

__all__ = [
    "bounded_repr",
    "bounded_text",
]


class _DiagnosticRenderer:
    MAX_DEPTH: int = 3
    MAX_INT_BITS: int = 512
    MAX_ITEMS: int = 4
    MAX_LENGTH: int = 96

    def render(self, value: object) -> str:
        """Render a value within a strict character budget.

        Args:
            value (object): The value to render.

        Returns:
            str: The bounded representation.
        """
        try:
            rendered = self._render(value=value, depth=0)
            return self._truncate(rendered)
        except Exception:  # ruff: ignore[blind-except] — diagnostics cannot mask errors
            rendered = f"<unrepresentable {self._type_name(value)}>"
            return self._truncate(rendered)

    def render_text(self, value: object) -> str:
        """Render strings without quotes and other values diagnostically.

        Args:
            value (object): The value to render.

        Returns:
            str: The bounded text.
        """
        if not isinstance(value, str):
            return self.render(value)
        try:
            return self._truncate(value)
        except Exception:  # ruff: ignore[blind-except] — diagnostics cannot mask errors
            return self.render(value)

    def _render(  # ruff: ignore[too-many-return-statements]
        self,
        *,
        value: object,
        depth: int,
    ) -> str:
        if depth >= self.MAX_DEPTH:
            return f"<{self._type_name(value)} ...>"
        if value is None or isinstance(value, bool):
            return repr(value)
        if isinstance(value, int):
            return self._render_int(value)
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, str | bytes):
            return self._render_string(value)
        if isinstance(value, list | tuple | set | frozenset):
            return self._render_collection(value=value, depth=depth)
        if isinstance(value, dict):
            return self._render_mapping(value=value, depth=depth)
        return repr(value)

    def _render_int(self, value: int) -> str:
        bit_length = int.bit_length(value)
        if bit_length > self.MAX_INT_BITS:
            return f"<int with {bit_length} bits>"
        return repr(value)

    def _render_string(self, value: str | bytes) -> str:
        preview_length = self.MAX_LENGTH // 2
        if isinstance(value, str):
            value_length = str.__len__(value)  # ruff: ignore[unnecessary-dunder-call]
            preview = str.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
                value,
                slice(0, preview_length),
            )
        else:
            value_length = bytes.__len__(  # ruff: ignore[unnecessary-dunder-call]
                value,
            )
            preview = bytes.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
                value,
                slice(0, preview_length),
            )
        rendered = repr(preview)
        if value_length > preview_length:
            rendered = f"{rendered}..."
        return rendered

    def _render_collection(
        self,
        *,
        value: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
        depth: int,
    ) -> str:
        opening, closing = self._collection_delimiters(value)
        parts = [
            self._render(value=item, depth=depth + 1)
            for item in islice(value, self.MAX_ITEMS)
        ]
        if len(value) > self.MAX_ITEMS:
            parts.append("...")
        return self._truncate(f"{opening}{', '.join(parts)}{closing}")

    def _render_mapping(self, *, value: dict[Any, Any], depth: int) -> str:
        parts = [
            (
                f"{self._render(value=key, depth=depth + 1)}: "
                f"{self._render(value=item, depth=depth + 1)}"
            )
            for key, item in islice(value.items(), self.MAX_ITEMS)
        ]
        if len(value) > self.MAX_ITEMS:
            parts.append("...")
        return self._truncate(f"{{{', '.join(parts)}}}")

    def _type_name(self, value: object) -> str:
        return self._truncate(type(value).__name__)

    def _truncate(self, value: str) -> str:
        value_length = str.__len__(value)  # ruff: ignore[unnecessary-dunder-call]
        if value_length <= self.MAX_LENGTH:
            return str.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
                value,
                slice(0, value_length),
            )
        preview = str.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
            value,
            slice(0, self.MAX_LENGTH - 3),
        )
        return f"{preview}..."

    @staticmethod
    def _collection_delimiters(
        value: list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any],
    ) -> tuple[str, str]:
        if isinstance(value, list):
            return ("[", "]")
        if isinstance(value, tuple):
            return ("(", ")")
        if isinstance(value, set):
            return ("{", "}")
        return ("frozenset({", "})")


def bounded_repr(value: object) -> str:
    """Return a bounded, exception-safe representation of a value.

    Args:
        value (object): The value to represent.

    Returns:
        str: The bounded representation.
    """
    return _DiagnosticRenderer().render(value)


def bounded_text(value: object) -> str:
    """Return bounded text without invoking arbitrary string conversion.

    Args:
        value (object): The value to render as text.

    Returns:
        str: The bounded text.
    """
    return _DiagnosticRenderer().render_text(value)
