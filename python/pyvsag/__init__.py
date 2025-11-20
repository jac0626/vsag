
from ._pyvsag import *  # noqa: F401,F403
from ._version import __version__

__all__ = []  # populated by the C-extension's __all__ if it defines one
try:
    from ._pyvsag import __all__ as _pyvsag_all  # type: ignore
    if isinstance(_pyvsag_all, (list, tuple)):
        __all__.extend(_pyvsag_all)
except Exception:
    pass

__all__.append("__version__")
