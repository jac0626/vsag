
import os
import sys

_cur_file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_cur_file_dir)

from _pyvsag import *  # noqa: F401,F403

try:
    from ._version import __version__
except Exception:  # pragma: no cover - best effort fallback
    _override = os.getenv("PYVSAG_OVERRIDE_VERSION")
    if _override:
        __version__ = _override  # type: ignore
    else:
        _metadata = None
        try:
            import importlib.metadata as _metadata  # type: ignore
        except ImportError:  # Python <3.8
            try:
                import importlib_metadata as _metadata  # type: ignore
            except ImportError:
                _metadata = None

        if _metadata is not None:
            try:
                __version__ = _metadata.version("pyvsag")  # type: ignore
            except Exception:
                __version__ = "0.0.0"  # type: ignore
        else:
            __version__ = "0.0.0"  # type: ignore
