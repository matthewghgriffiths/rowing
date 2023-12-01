import streamlit as st
import io 

import logging 

import numpy as np
import pandas as pd 

import plotly.graph_objects as go

from rowing.analysis import geodesy, splits, app, telemetry
from rowing import utils

logger = logging.getLogger(__name__)



st.set_page_config(
    page_title="Peach Telemetry Analysis",
    layout='wide'
)
"""
# Peach Telemetry processing
"""

with st.expander("Upload Telemetry Data"):
    tabs = st.tabs([
        "Upload csvs", "Upload xlsx", 
    ])
    telemetry_data = {}
    with tabs[0]:
        uploaded_files = st.file_uploader(
            "Upload All Data Export from PowerLine", 
            accept_multiple_files=True
        )
        for file in uploaded_files:
            telemetry_data[
                file.name.rsplit(".", 1)[0]
            ] = telemetry.parse_powerline_text_data(
                file.read().decode("utf-8"))

    gps_data = {
        k: split_data['positions'] for k, split_data in telemetry_data.items()
    }

with st.expander("Landmarks"):
    set_landmarks = app.set_landmarks()
    locations = set_landmarks.set_index(["location", "landmark"])

if not telemetry_data:
    st.write("No data uploaded")
    st.stop()

with st.expander("Show map"):
    app.draw_gps_data(gps_data, locations)

with st.spinner("Processing Crossing Times"):
    crossing_times = app.get_crossing_times(gps_data, locations=locations)
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
        avg_telem, interval_telem = {}, {}
        for piece, timestamps in piece_data['Timestamp'].iterrows():
            name = piece[1]
            power = telemetry_data[name][
                'power'
            ].sort_values("Time").reset_index(drop=True)
            avgP, intervalP = splits.get_interval_averages(
                power.drop("Time", axis=1), power.Time, timestamps
            )
            for k in avgP.columns.remove_unused_levels().levels[0]:
                avg_telem.setdefault(k, {})[name] = avgP[k].T
            for k in intervalP.columns.remove_unused_levels().levels[0]:
                interval_telem.setdefault(k, {})[name] = intervalP[k].T

        for k, data in avg_telem.items():
            piece_data[f"Average {k}"] = pd.concat(
                data, names=['file', 'position'])
        for k, data in interval_telem.items():
            piece_data[f"Interval {k}"] = pd.concat(
                data, names=['file', 'position'])

        app.show_piece_data(piece_data)

