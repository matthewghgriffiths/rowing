from pathlib import Path
import sys
import os

import streamlit as st

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent)

try:
    import rowing
    from rowing import app, world_rowing
except ImportError:

    st.write(LIBPATH)

    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    print(sys.path)

    st.write(sys.path)

    import rowing
    from rowing import app, world_rowing
