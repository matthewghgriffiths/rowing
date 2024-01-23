
import os
from pathlib import Path
import logging


import click
import streamlit
from streamlit.web import cli

from rowing.app import inputs, plots, select, state, threads

DIR = Path(__file__).absolute().parent
PACKAGE_DIR = (DIR / "../../..").resolve()

LOG_LEVELS = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.WARNING,
    "debug": logging.DEBUG
}


def get_working_directory(folder="app/world_rowing", start_dir=DIR):
    start_dir = Path(start_dir)
    for i in range(4):
        print(f"{start_dir=}")
        work_dir = start_dir / folder
        if os.path.exists(work_dir):
            print(work_dir)
            return work_dir

        start_dir = start_dir.parent

    raise FileNotFoundError("could not find 'app/world_rowing/main.py")


@click.command()
@click.option("--dir", show_default=True, default=str(DIR), help="folder to search for app")
@click.option("--application", show_default=True, default="world_rowing_app", help="name of folder with main.py")
@click.option("--log-level", show_default=True, type=click.Choice(LOG_LEVELS))
def main(dir, application, log_level):
    if log_level:
        level = LOG_LEVELS.get(log_level)
        if level:
            logging.getLogger().setLevel(level)

    work_dir = get_working_directory(application, dir)

    streamlit._is_running_with_streamlit = True
    flag_options = {
        "global.developmentMode": False
    }
    os.chdir(work_dir)
    cli.bootstrap.load_config_options(flag_options=flag_options)
    cli._main_run("home.py", flag_options)


if __name__ == "__main__":
    main()
