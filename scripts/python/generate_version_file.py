#!/usr/bin/env python3
"""
Utility to materialize `python/pyvsag/_version.py`.

The script mirrors the GitHub workflow in commit 09f64f4 by calling
`setuptools_scm.get_version()` and persisting the resolved version so that
subsequent builds (including CMake and cibuildwheel) can import pyvsag
without requiring setuptools-scm at runtime.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "python" / "pyvsag" / "_version.py"


def resolve_version(explicit: str | None, fallback: str) -> str:
    """Resolve the version string using, in order, an explicit override,
    the PYVSAG_OVERRIDE_VERSION environment variable, setuptools_scm, and
    finally the provided fallback."""
    if explicit:
        return explicit

    env_override = os.getenv("PYVSAG_OVERRIDE_VERSION")
    if env_override:
        return env_override

    try:
        from setuptools_scm import get_version  # type: ignore

        return get_version(root=str(REPO_ROOT), relative_to=__file__)
    except Exception as exc:  # pragma: no cover - best effort fallback
        print(
            f"[generate_version_file] Warning: failed to infer version via "
            f"setuptools_scm ({exc}). Falling back to '{fallback}'.",
            file=sys.stderr,
        )
        return fallback


def write_version_file(output: Path, version: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Target file to write (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Explicit version string to write. "
        "Overrides environment and auto detection.",
    )
    parser.add_argument(
        "--fallback",
        type=str,
        default="0.0.0",
        help="Version to use if auto detection fails.",
    )
    args = parser.parse_args(argv)

    version = resolve_version(args.version, args.fallback)
    write_version_file(args.output, version)
    print(f"[generate_version_file] Wrote {args.output} with version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

