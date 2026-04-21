# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from tools.rampart_lint.rule import is_dunder


class TestIsDunder:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("__init__", True),
            ("__aenter__", True),
            ("__eq__", True),
            ("____", True),
            ("_private", False),
            ("__leading", False),
            ("trailing__", False),
            ("normal", False),
            ("", False),
        ],
    )
    def test_detection(self, name: str, expected: bool) -> None:  # noqa: FBT001
        assert is_dunder(name) is expected
