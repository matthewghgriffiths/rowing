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
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    import rowing
    from rowing import app, world_rowing


def main():
    st.set_page_config(
        page_title="Other",
        layout='wide'
    )


if __name__ == "__main__":
    main()
