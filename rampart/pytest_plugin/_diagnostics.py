# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded diagnostics for pytest plugin validation boundaries."""

from __future__ import annotations

from types import GetSetDescriptorType
from typing import cast

from rampart.common.text import escape_terminal_controls

__all__ = [
    "bounded_repr",
    "bounded_text",
    "safe_type_name",
]


class _DiagnosticRenderer:
    MAX_INT_BITS: int = 512
    MAX_LENGTH: int = 192
    PREVIEW_LENGTH: int = 48

    def render(self, value: object) -> str:
        """Render exact safe builtins or a fixed type placeholder.

        Args:
            value (object): The value to render.

        Returns:
            str: The escaped, bounded diagnostic representation.
        """
        try:
            rendered = self._render_exact(value)
        except Exception:  # ruff: ignore[blind-except] — diagnostics cannot mask errors
            rendered = "<object>"
        return self._finish(rendered)

    def render_text(self, value: object) -> str:
        """Render exact strings without quotes and other values opaquely.

        Args:
            value (object): The value to render.

        Returns:
            str: The escaped, bounded diagnostic text.
        """
        if type(value) is not str:
            return self.render(value)
        try:
            value_length = str.__len__(  # ruff: ignore[unnecessary-dunder-call]
                value,
            )
            preview = str.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
                value,
                slice(0, self.MAX_LENGTH),
            )
            if value_length > self.MAX_LENGTH:
                preview = f"{preview}..."
            return self._finish(preview)
        except Exception:  # ruff: ignore[blind-except] — diagnostics cannot mask errors
            return "<str>"

    def type_name(self, value: object) -> str:
        """Return an escaped, bounded non-polymorphic type name.

        Args:
            value (object): The value whose exact type should be named.

        Returns:
            str: The safe type name or a constant fallback.
        """
        try:
            value_type = type(value)
            # Bypass custom metaclass attribute hooks with type's built-in descriptor.
            name_descriptor = cast(
                "GetSetDescriptorType",
                type.__dict__["__name__"],
            )
            name = GetSetDescriptorType.__get__(
                name_descriptor,
                value_type,
                type,
            )
            if type(name) is not str:
                return "object"
            return self._finish(name)
        except Exception:  # ruff: ignore[blind-except] — hostile metaclasses
            return "object"

    def _render_exact(  # ruff: ignore[too-many-return-statements]
        self,
        value: object,
    ) -> str:
        value_type = type(value)
        if value_type is type(None):
            return "None"
        if value_type is bool:
            return "True" if value is True else "False"
        if value_type is int:
            return self._render_int(cast("int", value))
        if value_type is float:
            return float.__repr__(value)  # ruff: ignore[unnecessary-dunder-call]
        if value_type is str:
            return self._render_str(cast("str", value))
        if value_type is bytes:
            return self._render_bytes(cast("bytes", value))
        container_length = self._exact_container_length(value)
        if container_length is not None:
            return (
                f"<{self.type_name(value)} "
                f"len={int.__repr__(container_length)}>"  # ruff: ignore[unnecessary-dunder-call]
            )
        return f"<{self.type_name(value)}>"

    def _render_int(self, value: int) -> str:
        bit_length = int.bit_length(value)
        if bit_length > self.MAX_INT_BITS:
            return f"<int with {bit_length} bits>"
        return int.__repr__(value)  # ruff: ignore[unnecessary-dunder-call]

    def _render_str(self, value: str) -> str:
        value_length = str.__len__(value)  # ruff: ignore[unnecessary-dunder-call]
        preview = str.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
            value,
            slice(0, self.PREVIEW_LENGTH),
        )
        rendered = str.__repr__(preview)  # ruff: ignore[unnecessary-dunder-call]
        return f"{rendered}..." if value_length > self.PREVIEW_LENGTH else rendered

    def _render_bytes(self, value: bytes) -> str:
        value_length = bytes.__len__(value)  # ruff: ignore[unnecessary-dunder-call]
        preview = bytes.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
            value,
            slice(0, self.PREVIEW_LENGTH),
        )
        rendered = bytes.__repr__(preview)  # ruff: ignore[unnecessary-dunder-call]
        return f"{rendered}..." if value_length > self.PREVIEW_LENGTH else rendered

    @staticmethod
    def _exact_container_length(value: object) -> int | None:
        value_type = type(value)
        if value_type is list:
            return list.__len__(  # ruff: ignore[unnecessary-dunder-call]
                cast("list[object]", value),
            )
        if value_type is tuple:
            return tuple.__len__(  # ruff: ignore[unnecessary-dunder-call]
                cast("tuple[object, ...]", value),
            )
        if value_type is dict:
            return dict.__len__(  # ruff: ignore[unnecessary-dunder-call]
                cast("dict[object, object]", value),
            )
        if value_type is set:
            return set.__len__(  # ruff: ignore[unnecessary-dunder-call]
                cast("set[object]", value),
            )
        if value_type is frozenset:
            return frozenset.__len__(  # ruff: ignore[unnecessary-dunder-call]
                cast("frozenset[object]", value),
            )
        return None

    def _finish(self, value: str) -> str:
        raw_length = str.__len__(value)  # ruff: ignore[unnecessary-dunder-call]
        raw_preview = str.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
            value,
            slice(0, self.MAX_LENGTH),
        )
        escaped = escape_terminal_controls(raw_preview)
        if raw_length > self.MAX_LENGTH:
            escaped = f"{escaped}..."
        escaped_length = str.__len__(  # ruff: ignore[unnecessary-dunder-call]
            escaped,
        )
        if escaped_length <= self.MAX_LENGTH:
            return escaped
        preview = str.__getitem__(  # ruff: ignore[unnecessary-dunder-call]
            escaped,
            slice(0, self.MAX_LENGTH - 3),
        )
        return f"{preview}..."


def bounded_repr(value: object) -> str:
    """Return a bounded exact-builtins-only representation.

    Args:
        value (object): The value to represent.

    Returns:
        str: The escaped, bounded representation.
    """
    return _DiagnosticRenderer().render(value)


def bounded_text(value: object) -> str:
    """Return bounded text without invoking arbitrary string conversion.

    Args:
        value (object): The value to render as text.

    Returns:
        str: The escaped, bounded text.
    """
    return _DiagnosticRenderer().render_text(value)


def safe_type_name(value: object) -> str:
    """Return a non-polymorphic escaped and bounded type name.

    Args:
        value (object): The value whose exact type should be named.

    Returns:
        str: The safe type name.
    """
    return _DiagnosticRenderer().type_name(value)
