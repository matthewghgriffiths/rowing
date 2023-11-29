import streamlit as st
import io 

import logging 

import numpy as np
import pandas as pd 

from rowing.analysis import files, splits, geodesy
from rowing import utils

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GPX",
    layout='wide'
)
"""
# GPX data processing
"""

uploaded_files = st.file_uploader(
    "Upload GPX files", accept_multiple_files=True
)

def parse_gpx(file):
    return files.parse_gpx_data(files.gpxpy.parse(file))

def download_csv(file_name, df, label=":inbox_tray: Download data as csv", **kwargs):
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


with st.expander("Piece selecter"):
    piece_dates = np.sort(all_crossing_times.dt.date.unique())

    cols = st.columns(3)
    with cols[0]:
        select_date = st.selectbox(
            "select piece date", 
            piece_dates, 
            index=piece_dates.size - 1, 
        )
    sel_times = all_crossing_times[
        all_crossing_times.dt.date == select_date
    ]
    landmarks = sel_times.index.levels[3]
    with cols[1]:
        start_landmark = st.selectbox(
            "select start landmark", 
            landmarks, 
        )
    with cols[2]:
        finish_landmark = st.selectbox(
            "select finish landmark", 
            landmarks, 
        )

    start_times = sel_times.xs(start_landmark, level=3).droplevel(-1)
    finish_times = sel_times.xs(finish_landmark, level=3).droplevel(-1)
    times = pd.concat({
        "Elapsed time": sel_times - start_times,
        "Time left": finish_times - sel_times,
        "Time": sel_times, 
    }, axis=1)
    piece_data = times[
        times.notna().all(axis=1)
        & (times['Time left'].dt.total_seconds() >= 0)
        & (times["Elapsed time"].dt.total_seconds() >= 0)
    ].reset_index("distance").unstack()
    legs = piece_data.index
    startfinish = pd.concat({
        "Start time": start_times.loc[legs], 
        "Finish Time": finish_times.loc[legs]
    }, axis=1)
    piece_data.index = pd.MultiIndex.from_frame(
        start_times.loc[legs].rename("Start Time").reset_index()[
            ["Start Time", "file", "location", "leg"]
        ]
    )
    piece_data = piece_data.sort_index(level=0)
    legs = piece_data.index.droplevel(0)

    piece_distances = (
        piece_data.distance 
        - piece_data.distance[start_landmark].values[:, None]
    )
    piece_time = piece_data['Elapsed time']
    avg_split = (piece_time * 0.5 / piece_distances).fillna(pd.Timedelta(0))
    interval_split = (
        piece_time.diff(axis=1) * 0.5 / piece_distances.diff(axis=1)
    ).fillna(pd.Timedelta(0))
    
    st.dataframe(
        startfinish
    )
    tabs = st.tabs([
        "Elapsed Time", 
        "Distance Travelled", 
        "Average Split", 
        "Interval Split", 
        "Timestamp"
    ])
    with tabs[0]:
        st.dataframe(
            piece_time.applymap(
                utils.format_timedelta, #hours=True
            )
        )
    with tabs[1]:
        st.dataframe(
            piece_distances
        )

    with tabs[2]:
        st.dataframe(
            interval_split.applymap(
                utils.format_timedelta, #hours=True
            )
        )

    with tabs[3]:
        st.dataframe(
            avg_split.applymap(
                utils.format_timedelta, #hours=True
            )
        )
    with tabs[4]:
        st.dataframe(piece_data['Time'])

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

with st.spinner("Processing fastest times"):
    best_times, errors = utils.map_concurrent(
        splits.find_all_best_times, 
        gpx_data, singleton=True, 
    )

with st.expander("Fastest times"):
    tabs = st.tabs(best_times)
    for tab, (name, times) in zip(tabs, best_times.items()):
        show_times = times + pd.Timestamp(0)
        print(times.dtypes)
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
            download_csv(
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
            timings.index.names = "", 'leg', 'landmark', 'distance'
            timings.columns.names = "", 'leg', 'landmark', 'distance'
            upload_timings = timings.applymap(
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
