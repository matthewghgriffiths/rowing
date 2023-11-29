import streamlit as st

import logging 

import numpy as np 
import pandas as pd 

from tqdm.auto import tqdm

from rowing.analysis import files, splits, geodesy
from rowing import utils

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GPS",
    layout='wide'
)
"""
# GPS data processing
"""

uploaded_files = st.file_uploader(
    "Upload GPX files", accept_multiple_files=True
)

def parse_gpx(file):
    return files.parse_gpx_data(files.gpxpy.parse(file))

def download_csv(file_name, df, label="Download data as CSV", **kwargs):
    st.download_button(
        label=label, 
        file_name=file_name,
        data=df.to_csv().encode("utf-8"), 
        mime="text/csv",
        **kwargs, 
    )

gpx_data, errors = utils.map_concurrent(
    parse_gpx, 
    {file.name.rsplit(".", 1)[0]: file for file in uploaded_files}, 
    singleton=True
)

if not gpx_data:
    st.write("No data uploaded")
    st.stop()

with st.spinner("Processing Crossing Times"):
    crossing_times, errors = utils.map_concurrent(
        splits.find_all_crossing_times, gpx_data, singleton=True
    )
    all_crossing_times = pd.concat(crossing_times, names=['file'])


with st.expander("All Crossing times"):
    show_times = pd.concat({
        "date": all_crossing_times.dt.normalize(), 
        "time": all_crossing_times,
    }, axis=1)
    st.dataframe(
        show_times, 
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "time": st.column_config.TimeColumn(
                "Time", format="hh:mm:ss.SS"
            )
        }
    )
    
    download_csv("all-crossings.csv", show_times)

with st.expander("Individual Crossing Times"):
    tabs = st.tabs(crossing_times)
    for tab, (name, crossings) in zip(tabs, crossing_times.items()):
        with tab:
            show_crossings = pd.concat({
                "date": crossings.dt.normalize(), 
                "time": crossings,
            }, axis=1)
            st.dataframe(
                show_crossings, 
                column_config={
                    "date": st.column_config.DateColumn("Date"),
                    "time": st.column_config.TimeColumn(
                        "Time", format="hh:mm:ss.SS"
                    )
                }
            )
            download_csv(f"{name}-crossings.csv", show_crossings)


with st.spinner("Processing split timings"):
    location_timings, errors = utils.map_concurrent(
        splits.get_location_timings, gpx_data, singleton=True
    )

with st.expander("Landmark timings"):
    tabs = st.tabs(location_timings)
    for tab, (name, timings) in zip(tabs, location_timings.items()):
        with tab:
            timings.index.names = "", 'leg', 'landmark', 'distance'
            timings.columns.names = "", 'leg', 'landmark', 'distance'
            upload_timings = timings.applymap(
                utils.format_timedelta, hours=True
            ).replace("00:00:00.00", "").T

            st.dataframe(upload_timings)

            st.download_button(
                label="Download data as CSV", 
                file_name=f"{name}-timings.csv",
                data=upload_timings.to_csv().encode("utf-8"), 
                mime="text/csv",
            )

