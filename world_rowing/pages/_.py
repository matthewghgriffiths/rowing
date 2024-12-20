from pathlib import Path
import sys
import os

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent.parent)

try:
    import rowing
except (ImportError, ModuleNotFoundError):
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    import rowing
