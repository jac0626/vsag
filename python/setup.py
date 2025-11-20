from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from setuptools import Extension, setup


PROJECT_DIR = Path(__file__).resolve().parent
VERSION_FILE = PROJECT_DIR / "pyvsag" / "_version.py"


def _write_version_file(version: str) -> None:
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def _read_version_file() -> Optional[str]:
    if not VERSION_FILE.exists():
        return None
    namespace: Dict[str, Any] = {}
    exec(VERSION_FILE.read_text(encoding="utf-8"), namespace)
    return namespace.get("__version__")


def _resolve_version() -> str:
    existing = _read_version_file()
    if existing:
        return existing

    override = os.getenv("PYVSAG_OVERRIDE_VERSION")
    if override:
        _write_version_file(override)
        return override

    try:
        from setuptools_scm import get_version  # type: ignore

        version = get_version(root=str(PROJECT_DIR.parent), relative_to=__file__)
    except Exception:
        version = "0.0.0"

    _write_version_file(version)
    return version


# All other configuration is in setup.cfg / pyproject.toml.
# This file is kept for the C-extension definition and to mirror the
# version-generation logic from commit 09f64f4.
setup(
    version=_resolve_version(),
    ext_modules=[Extension("example", sources=["example.c"])],
    zip_safe=False,
)
