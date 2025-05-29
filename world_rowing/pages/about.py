from pathlib import Path
import sys
import os

import streamlit as st

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent.parent)

try:
    import rowing
except (ImportError, ModuleNotFoundError):
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    import rowing


def main():
    st.set_page_config(
        page_title="About",
        layout='wide'
    )


if __name__ == "__main__":
    main()
