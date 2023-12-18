import streamlit as st
import io 
from functools import partial
from pathlib import Path 
import os
import sys
import zipfile 
import shutil

import logging 

import numpy as np
import pandas as pd 

import plotly.graph_objects as go
import plotly.express as px

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent)
realpaths = [os.path.realpath(p) for p in sys.path]
if LIBPATH not in realpaths:
    sys.path.append(LIBPATH)

from rowing.analysis import geodesy, splits, app, telemetry
from rowing import utils

logger = logging.getLogger("telemetry")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s', 
)


logger.info("telemetry")
st.set_page_config(
    page_title="Peach Telemetry Analysis",
    layout='wide'
)
"""
# Peach Telemetry processing
"""

@st.cache_data
def parse_excel(file, use_names=True):
    data = pd.read_excel(file, header=None)
    return telemetry.parse_powerline_excel(data, use_names=use_names)


with st.expander("Upload Telemetry Data"):
    use_names = st.checkbox("Use crew list", True)
    tabs = st.tabs([
        "Upload text", "Upload xlsx", "Upload csv", 
    ])
    telemetry_data = {}
    with tabs[0]:
        uploaded_files = st.file_uploader(
            "Upload All Data Export from PowerLine (tab separated)", 
            accept_multiple_files=True
        )
        st.write("Text should should be formatted like below (no commas!)")
        st.code(r""" 
            =====	File Info
            Serial #	Session	Filename	Start Time	TZBIAS	Location	Summary	Comments
            0000	156	<DATAPATH>\row000123-0000123D.peach-data	Sun, 01 Jan 2023 00:00:00	3600			
            =====	GPS Info
            Lat	Lon	UTC	PeachTime
            00.0000000000	00.0000000000	01 Jan 2023 00:00:00 (UTC)	00000
            =====	Crew Info
            ...
            """, None)
        uploaded_data = {
            file.name.rsplit(".", 1)[0]: file.read().decode("utf-8")
            for file in uploaded_files
        }
        data, errs = utils.map_concurrent(
            telemetry.parse_powerline_text_data, 
            uploaded_data, 
            singleton=True, 
            use_names=use_names,
        )
        telemetry_data.update(data)
        # for file in uploaded_files:
        #     telemetry_data[
        #         file.name.rsplit(".", 1)[0]
        #     ] = telemetry.parse_powerline_text_data(
        #         file.read().decode("utf-8"), use_names=use_names
        #     )
    with tabs[2]:
        uploaded_files = st.file_uploader(
            "Upload All Data Export from PowerLine (comma separated)", 
            accept_multiple_files=True
        )
        st.write("Text should should be formatted like below")
        st.code(""" 
            =====,File Info,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            Serial #,Session,Filename,Start Time,TZBIAS,Location,Summary,Comments,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            0000,000,<DATAPATH>\row000123-0000123D.peach-data,"Sun, 01 Jan 2023 00:00:00",0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            =====,GPS Info,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            Lat,Lon,UTC,PeachTime,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            00.00000000,00.00000000,Sun 01 Jan 2023 00:00:00 (UTC),00000,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
            ...
            """, None)
        uploaded_data = {
            file.name.rsplit(".", 1)[0]: file.read().decode("utf-8")
            for file in uploaded_files
        }
        data, errs = utils.map_concurrent(
            telemetry.parse_powerline_text_data, 
            uploaded_data, 
            singleton=True, 
            use_names=use_names,
            sep=',', 
        )
        telemetry_data.update(data)
        # for file in uploaded_files:
        #     telemetry_data[
        #         file.name.rsplit(".", 1)[0]
        #     ] = telemetry.parse_powerline_text_data(
        #         file.read().decode("utf-8"), sep=',', use_names=use_names)
    with tabs[1]:
        uploaded_files = st.file_uploader(
            "Upload Data Export from PowerLine", 
            accept_multiple_files=True
        )
        uploaded_data = {
            file.name.rsplit(".", 1)[0]: file
            for file in uploaded_files
        }
        data, errs = utils.map_concurrent(
            parse_excel, 
            uploaded_data, 
            singleton=True, 
            use_names=use_names,
        )
        telemetry_data.update(data)
        # for file in uploaded_files:
        #     k = file.name.rsplit(".", 1)[0]
        #     with st.spinner(f"Processing {k}"):
        #         telemetry_data[k] = parse_excel(file, use_names=use_names)


    gps_data = {
        k: split_data['positions'] for k, split_data in telemetry_data.items()
    }

with st.expander("Landmarks"):
    set_landmarks = app.set_landmarks()
    locations = set_landmarks.set_index(["location", "landmark"])

if not telemetry_data:
    st.write("No data uploaded")
    st.stop()

logger.warning("Show map")
with st.expander("Show map"):
    app.draw_gps_data(gps_data, locations)

logger.info("Crossing Times")
with st.spinner("Processing Crossing Times"):
    crossing_times = app.get_crossing_times(gps_data, locations=locations)
    all_crossing_times = pd.concat(crossing_times, names=['file'])

logger.info("All Crossing Times")
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
    
logger.info("Individual Crossing Times")
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

logger.info("Piece selecter")
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


tab_names = [
    'Angle 0.7 F', 
    'Angle Max F', 
    'Average Power', 
    'AvgBoatSpeed',
    'CatchSlip', 
    'Dist/Stroke', 
    'Drive Start T', 
    'Drive Time',
    'FinishSlip', 
    'Max Force PC', 
    'MaxAngle', 
    'MinAngle', 
    'Length', 
    'Effective', 
    'Rating',
    'Recovery Time', 
    'Rower Swivel Power', 
    'StrokeNumber', 
    'SwivelPower',
    # 'Time', 
    'Work PC Q1', 
    'Work PC Q2',
    'Work PC Q3', 
    'Work PC Q4'
]
logger.info("Plot data")
with st.expander("Plot data"):
    if piece_data:
        cols = st.columns(2)
        with cols[0]:
            window = st.number_input(
                "Select window to average over (s), set to 0 to remove smoothing",
                value=10, 
                min_value=0, 
                step=5, 
                # placeholder='', 
                # format="%d s",
                # step="0:00:10"
            )
        with cols[1]:
            height = st.number_input(
                "Set figures height", 
                100, 3000, 600
            )
        
        telemetry_figures = app.make_telemetry_figures(
            telemetry_data, piece_data, window=window, tab_names=tab_names
        )
        telem_tabs = dict(zip(tab_names, st.tabs(tab_names)))
        for col, tab in telem_tabs.items():
            with tab:
                for name in piece_data['Timestamp'].index.get_level_values(1):
                    fig = telemetry_figures[col, name]
                    fig.update_layout(
                        height=height, 
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                st.subheader("Interval Averages")
                interval_stats = piece_data[f"Interval {col}"]
                st.write(interval_stats)
        
                st.subheader("Piece Averages")
                interval_stats = piece_data[f"Average {col}"]
                st.write(interval_stats)


logger.info("Download data")
cols = st.columns(2)
with cols[0], st.spinner("Generating excel file"):
    xldata = io.BytesIO()
    with pd.ExcelWriter(xldata) as xlf:
        for name, data in piece_data.items():
            save_data = data.copy()
            for c, vals in data.items():
                if pd.api.types.is_datetime64_any_dtype(vals.dtype):
                    save_data[c] = vals.dt.tz_localize(None)
                elif pd.api.types.is_timedelta64_dtype(vals.dtype):
                    save_data[c] = vals.map(
                        partial(utils.format_timedelta, hours=True)
                    )

            # print(name)
            save_data.to_excel(xlf, name.replace("/", " per "))

    xldata.seek(0)
    st.download_button(
        ":inbox_tray: Download telemetry_piece_data.xlsx", 
        xldata,
        # type='primary', 
        file_name="telemetry_piece_data.xlsx",  
    )

with cols[1]:
    download_figures = st.multiselect(
        "Download Figures as", 
        options=['html', 'png', 'svg', 'pdf', 'jpg', 'webp'] ,
        default=[], 
    )
    for file_type in download_figures:
        import plotly.io as pio
        pio.templates.default = "plotly"
        with st.spinner(f"Creating {file_type} zip file"):
            zipdata = app.figures_to_zipfile(
                {f"{col}/{name}": fig for (col, name), fig in telemetry_figures.items()},
                file_type
            )
            st.download_button(
                f":inbox_tray: Download telemetry_figures_{file_type}.zip", 
                zipdata,
                # type='primary', 
                file_name=f"telemetry_figures_{file_type}.zip",  
            )