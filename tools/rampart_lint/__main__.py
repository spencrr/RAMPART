# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""CLI entry point for RAMPART's custom lint rules.

Usage::

    python -m tools.rampart_lint check rampart/
    python -m tools.rampart_lint list
    python -m tools.rampart_lint explain RAMPART001

Exit code 0 = clean, 1 = violations found, 2 = usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tools.rampart_lint.driver import RULES, check_file

if TYPE_CHECKING:
    from collections.abc import Callable


def _collect_paths(targets: list[Path]) -> list[Path]:
    """Expand CLI targets into a sorted list of ``.py`` files."""
    paths: list[Path] = []
    for p in targets:
        if p.is_file() and p.suffix == ".py":
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(p.rglob("*.py")))
        else:
            print(f"Skipping non-Python target: {p}", file=sys.stderr)
    return paths


def _cmd_check(args: argparse.Namespace) -> int:
    """Run lint checks on the given paths."""
    paths = _collect_paths(args.paths)
    all_errors: list[str] = []
    for path in paths:
        all_errors.extend(check_file(path))

    for err in all_errors:
        print(err)

    return 1 if all_errors else 0


def _cmd_list(_args: argparse.Namespace) -> int:
    """Print a summary table of all registered rules."""
    for rule in RULES:
        print(f"{rule.code}  {rule.description}")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    """Print detailed information about a single rule."""
    for rule in RULES:
        if rule.code == args.code:
            node_names = ", ".join(t.__name__ for t in rule.node_types)
            print(f"Code:        {rule.code}")
            print(f"Description: {rule.description}")
            print(f"Class:       {type(rule).__name__}")
            print(f"Node types:  {node_names}")
            return 0
    print(f"Unknown rule code: {args.code}", file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="rampart_lint",
        description="RAMPART custom lint rules (RAMPART001, RAMPART002, ...).",
    )
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="Lint files or directories.")
    check.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to lint.",
    )
    check.set_defaults(func=_cmd_check)

    list_cmd = sub.add_parser("list", help="List all registered rules.")
    list_cmd.set_defaults(func=_cmd_list)

    explain = sub.add_parser("explain", help="Show details for a rule code.")
    explain.add_argument(
        "code",
        type=str,
        help="Rule code (e.g. RAMPART001).",
    )
    explain.set_defaults(func=_cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args(args=argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return 2

    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
