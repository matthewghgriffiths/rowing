
import streamlit as st
import io
from pathlib import Path
import os
import sys
import warnings

import logging

import time
import urllib.parse

# import numpy as np
import pandas as pd

# import plotly.graph_objects as go
# import plotly.express as px

try:
    from rowing import utils
    from rowing.app import inputs
    from rowing.analysis import app, strava, garmin_app as garmin, splits
except ImportError:
    DIRPATH = Path(__file__).resolve().parent
    LIBPATH = str(DIRPATH.parent)
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    from rowing import utils
    from rowing.app import inputs
    from rowing.analysis import app, strava, garmin_app as garmin, splits


logger = logging.getLogger(__name__)


def main(state=None):
    state = state or {}
    data = state.pop("gpx_data", {})
    st.session_state.update(state)

    st.set_page_config(
        page_title="Rowing GPS Analysis",
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    st.title("Rowing GPS Analysis")
    with st.sidebar:
        if st.button("Reset State"):
            st.session_state.clear()
            st.cache_resource.clear()

    with st.expander("Load Data", expanded=True):
        if 'code' in st.query_params or 'strava' in st.query_params:
            strava_tab, gpx_tab, garmin_tab = st.tabs([
                "Load Strava Activities",
                "Upload GPX",
                "Connect Garmin",
            ])
        else:
            gpx_tab, garmin_tab, = st.tabs([
                "Upload GPX",
                "Connect Garmin",
            ])
            strava_tab = gpx_tab

    with gpx_tab:
        uploaded_files = st.file_uploader(
            "Upload GPX files",
            accept_multiple_files=True,
            type=['gpx'],
        )
        gpx_data, errors = utils.map_concurrent(
            app.parse_gpx,
            {file.name.rsplit(".", 1)[0]: file for file in uploaded_files},
            singleton=True,
        )

    with strava_tab:
        strava_data = strava.strava_app()
        if strava_data:
            gpx_data.update(strava_data)

    with garmin_tab:
        garmin_client = garmin.login(*st.columns(3))
        if garmin_client:
            activities_tab, stats_tab = st.tabs(
                ["Load Activities", "Load Health Stats"])
            with activities_tab:
                garmin_data = garmin.garmin_activities_app(garmin_client)
                if garmin_data:
                    gpx_data.update(garmin_data)

            with stats_tab:
                garmin.garmin_stats_app(garmin_client)

    gpx_data.update(data)

    return analyse_gps_data(gpx_data)


@st.fragment
def garmin_stats(garmin_client):
    if garmin_client:
        cols = st.columns(2)
        with cols[0]:
            date1 = st.date_input(
                "Select Date",
                key="Garmin Stats Select Date",
                value=pd.Timestamp.today() + pd.Timedelta("1d"),
                format='YYYY-MM-DD'
            )
        with cols[1]:
            date2 = st.date_input(
                "Range",
                key="Garmin Stats Range",
                value=pd.Timestamp.today() - pd.Timedelta("7d"),
                format='YYYY-MM-DD'
            )

        stats = garmin.get_garmin_sleep_stats(
            garmin_client.username, date1, date2
        )

        st.dataframe(stats)


@st.fragment
def analyse_gps_data(gpx_data):
    with st.expander("Landmarks"):
        set_landmarks = app.set_landmarks(gps_data=gpx_data)
        locations = set_landmarks.set_index(["location", "landmark"])

    if not gpx_data:
        st.write("No data uploaded")
        st.stop()
        raise st.runtime.scriptrunner.StopException()

    with st.expander("Show map"):
        app.draw_gps_data(gpx_data, locations)

    with st.spinner("Processing Crossing Times"):
        crossing_times = app.get_crossing_times(
            gpx_data, locations=locations)
        all_crossing_times = pd.concat(crossing_times, names=['name'])

    with st.expander("Piece selecter"):
        piece_information = app.select_pieces(
            all_crossing_times)

        if piece_information is None:
            st.write("No valid pieces could be found")
        else:
            options = []
            for d in gpx_data.values():
                options = d.columns.union(options)

            default = [
                'heart_rate',
                'cadence',
                'bearing',
            ]
            keep = st.multiselect(
                "Select data to keep",
                options=options,
                default=options.intersection(default),
                # default=
            )
            average_cols = ['time'] + keep
            piece_information['piece_data'].update(
                splits.get_pieces_interval_averages(
                    piece_information['piece_data']['Timestamp'],
                    {k: d[average_cols] for k, d in gpx_data.items()},
                    time='time'
                )
            )

            app.show_piece_data(piece_information['piece_data'])

    with st.spinner("Processing split timings"):
        location_timings = app.get_location_timings(
            gpx_data, locations=locations)

    with st.expander("Piece Timings"):
        timings_fragment(all_crossing_times, crossing_times, location_timings)

    with st.expander("Compare Piece Profile"):
        if piece_information:
            if keep:
                pace_tab, other_tab = st.tabs(['Pace Boat', 'Other'])
            else:
                pace_tab = st.container()

            piece_data = piece_information['piece_data']
            settings = st.popover("Figure settings")
            with settings:
                height = st.number_input(
                    "Set profile figure height",
                    100, None, 600, step=50,
                    key="height piece profile",
                )
            landmark_distances = piece_data['Distance Travelled'].mean()[
                piece_data['Total Distance'].columns]
            fig, time_behind = app.plot_pace_boat(
                piece_data,
                landmark_distances,
                gpx_data,
                input_container=settings,
                name='name',
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=height)
            with pace_tab:
                st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Processing fastest times"):
        best_times = app.get_fastest_times(gpx_data)

    with st.expander("Fastest times"):
        tabs = st.tabs(best_times)
        for tab, (name, times) in zip(tabs, best_times.items()):
            show_times = times + pd.Timestamp(0)
            with tab:
                st.dataframe(
                    show_times.iloc[::-1],
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
                    times.map(
                        utils.format_timedelta, hours=True
                    ).replace("00:00:00.00", "")
                )

    with st.spinner("Generating excel file"):
        excel_export_fragment(crossing_times, location_timings, best_times)


@st.fragment
def timings_fragment(all_crossing_times, crossing_times, location_timings):
    tab_all, *tabs = st.tabs(['All Crossing Times'] + list(crossing_times))
    with tab_all:
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

    for tab, (name, crossings) in zip(tabs, crossing_times.items()):
        with tab:
            show_crossings = pd.concat({
                "date": crossings.dt.normalize(),
                "time": crossings,
            }, axis=1)
            st.subheader("Crossing times")
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

    for tab, (name, timings) in zip(tabs, location_timings.items()):
        with tab:
            upload_timings = timings.droplevel(
                "location"
            ).droplevel("location", axis=1).rename_axis(
                ["", 'leg', 'landmark', 'distance']
            ).rename_axis(
                ["", 'leg', 'landmark', 'distance'], axis=1
            ).map(
                utils.format_timedelta, hours=True
            ).replace("00:00:00.00", "").T

            st.subheader("Timing Matrix")
            st.dataframe(upload_timings)
            app.download_csv(
                f"{name}-timings.csv",
                upload_timings,
            )


@st.fragment()
def excel_export_fragment(crossing_times, location_timings, best_times):
    if not any([crossing_times, location_timings, best_times]):
        print("no sheets")
        return

    xldata = io.BytesIO()
    with pd.ExcelWriter(xldata) as xlf, warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for name, crossings in crossing_times.items():
            crossings = crossings.rename("time").dt.tz_localize(None)
            crossings.to_frame().to_excel(
                xlf,
                sheet_name=f"{utils.safe_name(name)}-crossings"
            )

        for name, timings in location_timings.items():
            upload_timings = timings.droplevel(
                "location"
            ).droplevel("location", axis=1).rename_axis(
                ["", 'leg', 'landmark', 'distance']
            ).rename_axis(
                ["", 'leg', 'landmark', 'distance'], axis=1
            ).map(
                utils.format_timedelta, hours=True
            ).replace("00:00:00.00", "").T
            upload_timings.to_excel(
                xlf,
                sheet_name=f"{utils.safe_name(name)}-timings"
            )

        for name, times in best_times.items():
            times.map(
                utils.format_timedelta, hours=True
            ).replace("00:00:00.00", "").to_excel(
                xlf, f"{utils.safe_name(name)}-fastest"
            )

    xldata.seek(0)
    st.download_button(
        ":inbox_tray: Download all: GPS_data.xlsx",
        xldata,
        # type='primary',
        file_name="GPS_data.xlsx",
    )


if __name__ == "__main__":
    main()
