import streamlit as st
import io 
from pathlib import Path 
import os
import sys

import logging 

import numpy as np
import pandas as pd 

import plotly.graph_objects as go

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent)
realpaths = [os.path.realpath(p) for p in sys.path]
if LIBPATH not in realpaths:
    sys.path.append(LIBPATH)

from rowing.analysis import geodesy, splits, app
from rowing import utils

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Rowing GPX",
    layout='wide'
)
"""
# GPX data processing
"""

uploaded_files = st.file_uploader(
    "Upload GPX files", accept_multiple_files=True
)

with st.expander("Landmarks"):
    set_landmarks = app.set_landmarks()
    locations = set_landmarks.set_index(["location", "landmark"])

gpx_data, errors = utils.map_concurrent(
    app.parse_gpx, 
    {file.name.rsplit(".", 1)[0]: file for file in uploaded_files}, 
    singleton=True, 
)

if not gpx_data:
    st.write("No data uploaded")
    st.stop()

with st.expander("Show map"):
    app.draw_gps_data(gpx_data, locations)

with st.spinner("Processing Crossing Times"):
    crossing_times = app.get_crossing_times(gpx_data, locations=locations)
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
    app.download_csv("all-crossings.csv", show_times)


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
            app.download_csv(f"{name}-crossings.csv", show_crossings)


with st.expander("Piece selecter"):
    piece_data = app.select_pieces(all_crossing_times)
    if piece_data is None:
        st.write("No valid pieces could be found")
    else:
        app.show_piece_data(piece_data)

with st.spinner("Processing split timings"):
    location_timings = app.get_location_timings(gpx_data, locations=locations)

with st.expander("Landmark timings"):
    tabs = st.tabs(location_timings)
    for tab, (name, timings) in zip(tabs, location_timings.items()):
        with tab:
            upload_timings = timings.droplevel(
                "location"
            ).droplevel("location", axis=1).rename_axis(
                ["", 'leg', 'landmark', 'distance']
            ).rename_axis(
                ["", 'leg', 'landmark', 'distance'], axis=1
            ).applymap(
                utils.format_timedelta, hours=True
            ).replace("00:00:00.00", "").T
            st.dataframe(upload_timings)
            app.download_csv(
                f"{name}-timings.csv",
                upload_timings,
            )

with st.spinner("Processing fastest times"):
    best_times = app.get_fastest_times(gpx_data)

with st.expander("Fastest times"):
    tabs = st.tabs(best_times)
    for tab, (name, times) in zip(tabs, best_times.items()):
        show_times = times + pd.Timestamp(0)
        with tab:
            st.dataframe(
                show_times, 
                column_config={
                    "split": st.column_config.TimeColumn(
                        "Split", format="m:ss.SS"
                    ),
                    "time": st.column_config.TimeColumn(
                        "Time", format="m:ss.SS"
                    )
                }
            )
            app.download_csv(
                f"{name}-fastest.csv", 
                times.applymap(
                    utils.format_timedelta, hours=True
                ).replace("00:00:00.00", "")
            )

with st.spinner("Generating excel file"):
    xldata = io.BytesIO()
    with pd.ExcelWriter(xldata) as xlf:
        for name, crossings in crossing_times.items():
            crossings = crossings.rename("time").dt.tz_localize(None)
            crossings.to_frame().to_excel(
                xlf, f"{name}-crossings"
            )

        for name, timings in location_timings.items():
            upload_timings = timings.droplevel(
                "location"
            ).droplevel("location", axis=1).rename_axis(
                ["", 'leg', 'landmark', 'distance']
            ).rename_axis(
                ["", 'leg', 'landmark', 'distance'], axis=1
            ).applymap(
                utils.format_timedelta, hours=True
            ).replace("00:00:00.00", "").T
            upload_timings.to_excel(
                xlf, f"{name}-timings"
            )

        for name, times in best_times.items():
            times.applymap(
                utils.format_timedelta, hours=True
            ).replace("00:00:00.00", "").to_excel(
                xlf, f"{name}-fastest"
            )

    xldata.seek(0)
    st.download_button(
        ":inbox_tray: Download all: GPS_data.xlsx", 
        xldata,
        # type='primary', 
        file_name="GPS_data.xlsx",  
    )