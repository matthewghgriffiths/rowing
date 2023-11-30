
import streamlit as st
import logging 

from rowing.analysis import splits, files
from rowing import utils


@st.cache_data
def parse_gpx(file):
    return files.parse_gpx_data(files.gpxpy.parse(file))

def download_csv(
        file_name, df, label=":inbox_tray: Download data as csv", csv_kws=None, **kwargs
    ):
    st.download_button(
        label=label, 
        file_name=file_name,
        data=df.to_csv(**(csv_kws or {})).encode("utf-8"), 
        mime="text/csv",
        **kwargs, 
    )

@st.cache_data
def get_crossing_times(gpx_data):
    crossing_times, errors = utils.map_concurrent(
        splits.find_all_crossing_times, gpx_data, singleton=True
    )
    if errors:
        logging.error(errors)
    return crossing_times

@st.cache_data
def get_location_timings(gpx_data):
    location_timings, errors = utils.map_concurrent(
        splits.get_location_timings, gpx_data, singleton=True
    )
    if errors:
        logging.error(errors)
    return location_timings

@st.cache_data
def get_fastest_times(gpx_data):
    best_times, errors = utils.map_concurrent(
        splits.find_all_best_times, 
        gpx_data, singleton=True, 
    )
    if errors:
        logging.error(errors)
    return best_times