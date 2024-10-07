
import streamlit as st
import io
from pathlib import Path
import os
import sys

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
    from rowing.analysis import app, strava, garmin_app, files
except ImportError:
    DIRPATH = Path(__file__).resolve().parent
    LIBPATH = str(DIRPATH.parent)
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    from rowing import utils
    from rowing.app import inputs
    from rowing.analysis import app, strava, garmin_app, files


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

    with garmin_tab:
        cols = st.columns(3)
        garmin_client = garmin_app.login(*cols)
        print(garmin_client)

    with strava_tab:
        client = strava.connect_client()
        if client is not None:
            athlete = client.get_athlete()

            cols = st.columns(3)
            with cols[0]:
                limit = st.number_input(
                    "How many activities to load, "
                    "set to 0 if selecting date range",
                    value=1,
                    min_value=0,
                    step=1
                )
            with cols[1]:
                date1 = st.date_input(
                    "Select Date",
                    value=pd.Timestamp.today() + pd.Timedelta("1d"),
                    format='YYYY-MM-DD'
                )
            with cols[2]:
                date2 = st.date_input(
                    "Range",
                    value=pd.Timestamp.today() - pd.Timedelta("7d"),
                    format='YYYY-MM-DD'
                )

            if limit:
                start = end = None
            else:
                start, end = sorted(pd.to_datetime(
                    [date1, date2]).to_pydatetime())
                limit = None

            activities = strava.get_activities(client.code, end, start, limit)
            activities['athlete'] = f"{athlete.firstname} {athlete.lastname}"
            activities['activity'] = activities['athlete'].str.cat(
                activities[['name', 'start_date_local']].astype(str),
                sep=' '
            )

            columns_order = [
                'activity',
                'sport_type',
                'start_date_local',
                'distance',
                'elapsed_time',
                'type',
                'description',
                'name',
                'average_cadence',
                'average_heartrate',
                'average_speed',
            ]
            # columns_order = columns_order + list(
            #     activities.columns.difference(col_order))

            sel_activities = inputs.filter_dataframe(
                activities,
                select_all=False,
                column_order=columns_order
            )
            strava_data = {
                activity.activity: strava.load_strava_activity(
                    client.code, activity.id
                )
                for _, activity in sel_activities.iterrows()
            }
            gpx_data.update(strava_data)
            if st.toggle("Download gpx data"):
                for activity, activity_data in strava_data.items():
                    st.download_button(
                        f":inbox_tray: Download: {activity}.gpx",
                        io.BytesIO(
                            files.make_gpx_track(
                                activity_data).to_xml().encode()
                        ),
                        # type='primary',
                        file_name=f"{activity}.gpx",
                    )

    gpx_data.update(data)

    return analyse_gps_data(gpx_data)


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

    with st.expander("All Crossing times"):
        all_crossing_times = pd.concat(crossing_times, names=['name'])
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
        piece_information = app.select_pieces(
            all_crossing_times)
        if piece_information is None:
            st.write("No valid pieces could be found")
        else:
            app.show_piece_data(piece_information['piece_data'])

    with st.spinner("Processing split timings"):
        location_timings = app.get_location_timings(
            gpx_data, locations=locations)

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
                ).map(
                    utils.format_timedelta, hours=True
                ).replace("00:00:00.00", "").T
                st.dataframe(upload_timings)
                app.download_csv(
                    f"{name}-timings.csv",
                    upload_timings,
                )

    with st.expander("Compare Piece Profile"):
        if piece_information:
            piece_data = piece_information['piece_data']

            cols = st.columns((2, 1))
            with cols[1]:
                height = st.number_input(
                    "Set profile figure height",
                    100, 3000, 600, step=50,
                    key="height piece profile",
                )
            fig, time_behind = app.plot_pace_boat(
                piece_data,
                piece_data['Distance Travelled'].mean()[
                    piece_data['Total Distance'].columns],
                gpx_data,
                input_container=cols[0],
                name='name',
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)

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
                    times.map(
                        utils.format_timedelta, hours=True
                    ).replace("00:00:00.00", "")
                )

    with st.spinner("Generating excel file"):
        xldata = io.BytesIO()
        with pd.ExcelWriter(xldata) as xlf:
            for name, crossings in crossing_times.items():
                n = name.replace(":", '')
                crossings = crossings.rename("time").dt.tz_localize(None)
                crossings.to_frame().to_excel(
                    xlf, sheet_name=f"{n}-crossings"
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
                    xlf, f"{n}-timings"
                )

            for name, times in best_times.items():
                times.map(
                    utils.format_timedelta, hours=True
                ).replace("00:00:00.00", "").to_excel(
                    xlf, f"{n}-fastest"
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
