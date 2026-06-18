"""Pure file-parsing helpers for the rowing apps.

These functions load GPX / PowerLine / Peach exports into the in-memory
structures the apps work with. They contain no Streamlit calls and no caching,
so they can be imported and unit-tested directly; ``rowing.analysis.app``
exposes thin ``@st.cache_data`` wrappers around them for the Streamlit layer.
"""

import logging

import pandas as pd

from rowing import utils
from rowing.analysis import files, peach, telemetry

logger = logging.getLogger(__name__)


def parse_gpx(file):
    return files.parse_gpx_data(files.gpxpy.parse(file))


def parse_telemetry_text(uploaded_files, use_names=True, sep="\t", with_timings=True):
    uploaded_data = {file.name.rsplit(".", 1)[0]: file.read().decode("utf-8") for file in uploaded_files}
    data, errs = utils.map_concurrent(
        telemetry.parse_powerline_text_data,
        uploaded_data,
        singleton=True,
        use_names=use_names,
        with_timings=with_timings,
        sep=sep,
    )
    if errs:
        for k, err in errs.items():
            raise err
        logging.error(errs)

    return data


def parse_telemetry_files(uploaded_files, use_names=True, with_timings=False):
    uploaded_data = {file.name.rsplit(".", 1)[0]: file for file in uploaded_files}
    data, errs = utils.map_concurrent(
        parse_file, uploaded_data, singleton=True, use_names=use_names, with_timings=with_timings
    )
    if errs:
        for k, err in errs.items():
            raise err
        logging.error(errs)

    return data


def parse_file(file, use_names=True, with_timings=True):
    filename, *endings = file.name.rsplit(".", 1)
    (ending,) = endings or ("",)
    ending = ending.lower()
    if ending == "csv":
        return parse_text_data(file, use_names=use_names, sep=",", with_timings=with_timings)
    elif ending in {"xlsx", "xls"}:
        return parse_excel(file, use_names=use_names, with_timings=with_timings)
    elif ending == "zip":
        return telemetry.load_zipfile(file)
    return parse_text_data(file, use_names=use_names, sep="\t", with_timings=with_timings)


def parse_text_data(file, use_names=True, sep="\t", with_timings=True):
    return telemetry.parse_powerline_text_data(
        file.read().decode("utf-8"), use_names=use_names, sep=sep, with_timings=with_timings
    )


def parse_excel(file, use_names=True, with_timings=True):
    data = pd.read_excel(file, header=None)
    return telemetry.parse_powerline_excel(data, use_names=use_names, with_timings=with_timings)


def parse_telemetry_excel(uploaded_files, use_names=True, with_timings=True):
    uploaded_data = {file.name.rsplit(".", 1)[0]: file for file in uploaded_files}
    data, errs = utils.map_concurrent(
        parse_excel,
        uploaded_data,
        singleton=True,
        use_names=use_names,
        with_timings=with_timings,
    )
    if errs:
        logging.error(errs)

    return data


def parse_telemetry_zip(uploaded_files):
    telem_data = {}
    for file in uploaded_files:
        telem_data.update(telemetry.load_zipfile(file))

    return telem_data


def parse_peach_data(file, index_file=None, use_names=True, with_timings=True):
    index_bytes = index_file.read() if index_file else None
    data = peach.PeachData.from_bytes(file.read(), index_bytes, file.name)
    return data.app_data(use_names, with_timings)


def parse_peach_data_files(uploaded_files, use_names=True, with_timings=True):
    uploaded = {tuple(file.name.rsplit(".", 1)): file for file in uploaded_files}
    uploaded_filenames = set(k for k, end in uploaded if end.lower() == "peach-data")
    uploaded_data = {k: (uploaded[k, "peach-data"], uploaded.get((k, "peach-data-index"))) for k in uploaded_filenames}
    data, errs = utils.map_concurrent(
        parse_peach_data,
        uploaded_data,
        use_names=use_names,
        with_timings=with_timings,
    )
    if errs:
        logging.error(errs)

    return data
