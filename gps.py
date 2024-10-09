
import dis
import streamlit as st
import io
import warnings

import logging

# import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from rowing import utils
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
            keep = []
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

    if piece_information:
        with st.expander("Compare Piece Profile"):
            plot_piece_profiles(
                piece_information, gpx_data, [app.PACE_TIME_COL] + keep)

    with st.expander("Timings Summary"):
        timings_fragment(
            all_crossing_times, crossing_times, gpx_data, piece_information, locations)


@st.fragment
def plot_piece_profiles(piece_information, gpx_data, keep=None):

    piece_data = piece_information['piece_data']

    piece_times = piece_data['Elapsed Time'].apply(
        lambda x: (x + pd.Timestamp(0)).dt.time
    )
    use_pieces = piece_times.reset_index()
    use_pieces = app.inputs.filter_dataframe(
        use_pieces,
        key='filter plot_piece_profiles',
        disabled=use_pieces.columns,
        filters=True,
        select_all=True,
        column_config={
            c: st.column_config.TimeColumn(format="m:ss")
            for c in piece_times.columns
        }
    ).set_index(piece_times.index.names).index

    landmark_distances = piece_data['Distance Travelled'].loc[
        use_pieces
    ].mean()[
        piece_data['Total Distance'].columns]
    pace_boat_time = piece_data[
        'Elapsed Time'].loc[use_pieces].max(1).min()

    aligned_data = app.align_pieces(
        gpx_data, piece_data,
        landmark_distances=landmark_distances,
        pace_boat_time=pace_boat_time,
        pieces=use_pieces
    )
    if aligned_data.empty:
        return

    settings = st.popover("Figure settings")
    with settings:
        height = st.number_input(
            "Set profile figure height",
            100, None, 600, step=50,
            key="height piece profile",
        )
        keep = st.multiselect(
            "Select data to plot",
            options=aligned_data.columns,
            default=aligned_data.columns.intersection(keep or [])
        )

    tabs = st.tabs(keep)
    for tab, c in zip(tabs, keep):
        plot_data = aligned_data[c].reset_index()
        fig = px.line(
            plot_data,
            x='distance', y=c,
            color='name',
            line_dash='leg',
        )
        fig.update_layout(height=height)
        with tab:
            c2 = st.selectbox(
                "Plot on right axis",
                key=f"{c}_plotright",
                index=None,
                options=keep
            )
            if c2:
                fig2 = px.line(
                    aligned_data[c2].reset_index(),
                    x='distance', y=c2,
                    color='name',
                    line_dash='leg',
                )
                fig2.update_traces(opacity=0.5)

                for tr in fig2.data:
                    tr.yaxis = 'y2'
                    tr.showlegend = False
                    fig.add_trace(tr)

                fig.update_layout({
                    "yaxis2": dict(
                        title=dict(text=c2),
                        side='right',
                        tickmode="sync",
                        overlaying="y",
                        autoshift=True,
                        automargin=True,
                    )
                })

            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)


@st.fragment
def timings_fragment(all_crossing_times, crossing_times, gpx_data, piece_information, locations):

    with st.spinner("Processing split timings"):
        location_timings = app.get_location_timings(
            gpx_data, locations=locations)

    with st.spinner("Processing fastest times"):
        best_times = app.get_fastest_times(gpx_data)

    piece_data = piece_information['piece_data']

    names = list(crossing_times)
    data_names = list(piece_data)
    tab_all, *name_tabs = st.tabs(['All Crossing Times'] + names + data_names)
    tabs, data_tabs = name_tabs[:len(names)], name_tabs[len(names):]
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

    for tab, (name, times) in zip(tabs, best_times.items()):
        show_times = times + pd.Timestamp(0)
        with tab:
            st.subheader("Best Times")
            order = show_times.groupby(
                'length').time.min().sort_values(ascending=False)
            st.dataframe(
                show_times.loc[order.index],
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

    app.show_piece_data(piece_data, data_tabs)

    st.divider()

    with st.spinner("Generating excel file"):
        excel_export_fragment(
            crossing_times, location_timings, best_times, piece_information)


@st.fragment()
def excel_export_fragment(crossing_times, location_timings, best_times, piece_information):
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

        for key, data in piece_information['piece_data'].items():
            data = data.copy()
            for c, col in data.items():
                if pd.api.types.is_timedelta64_dtype(col.dtype):
                    data[c] = col.map(utils.format_timedelta)

            data.to_excel(xlf, sheet_name=key)

    xldata.seek(0)
    st.download_button(
        ":inbox_tray: Download all: GPS_data.xlsx",
        xldata,
        # type='primary',
        file_name="GPS_data.xlsx",
    )


if __name__ == "__main__":
    main()
