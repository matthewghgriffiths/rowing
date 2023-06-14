
import os
from pathlib import Path


import click
import streamlit 
from streamlit.web import cli

DIR = Path(__file__).absolute()
PACKAGE_DIR = (DIR / "../../..").resolve()


@click.command()
def main():
    streamlit._is_running_with_streamlit = True

    os.chdir(PACKAGE_DIR)
    cli.bootstrap.load_config_options(flag_options={
        "global.developmentMode": False
    })
    cli._main_run("worldrowing_app.py", ())


if __name__ == "__main__":
    main()
