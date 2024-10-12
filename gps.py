
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
from rowing.analysis import app, strava, garmin, splits
from rowing.world_rowing.fields import is_timedelta64_dtype


DEFAULT_FACETS = [
    'velocity_smooth',
    'heart_rate',
    'cadence',
    'split',
    'bearing',
]

logger = logging.getLogger(__name__)

HELP_TEXT = """## How to use

You can upload gpx files directly in the 'Upload GPX' tab, 

Alternatively you can connect your Strava or Garmin account. 
Your last activity will be automatically loaded. 
More activities can be loaded by increasing the 'limit'. 
Alternative you can search for activities within a certain range.

To remove log-out and remove all data associated with your account click 
the logout button.

- [Landmarks](#landmarks) shows a map of the landmarks and activities, 
it is possible to add landmarks from the pieces
- [Piece selecter](#piece-selecter) allows the 
start and end of the piece to be defined
- [Plot Piece Profile](#plot-piece-profile) shows plots of different
facets of data over the piece. Two facets of data (e.g. heart rate and time ahead)
can be displayed at the same time. To help visualise changes in pace, 
'Time/Distance ahead of Pace Boat' is provided as a facet. 
The fastest time for the piece in the data is used to set the speed of the pace boat, 
the relative time/distance ahead/behind a pace boat going at this speed. 
It is useful to note that travelling at a constant speed means that the trace will be straight line
horizontal if going at the same speed of the pace boat, slanted up if faster and down if slower.
- [Timings summary](#timings-summary) shows the summary of timings 
extracted from the activities, these tables can be downloaded as an 
excel spreadsheet. Included are all, 
  - crossing times for when an activity passed a landmark, 
  - timings matrix to record how much time passed between each crossing times, 
  - best times to cover a variety of different distances during each outing, and
  - information about subsegments of the selected piece, for example, 
  Interval split records the average split to cover a subsegment, 
  Average Split records the average split up to the listed landmark,
  if using data from Strava/Garmin average stroke rate (Cadence) and heart rate
  may also be displayed if available.

"""


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

        st.divider()
        st.button("Logout", on_click=clear_state)
        if st.toggle("Show help", True):
            st.markdown(HELP_TEXT)

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
        garmin_client = garmin.login(*st.columns(3))
        uploaded_fits = st.file_uploader(
            "(optional) Upload .fit files",
            type='fit',
            accept_multiple_files=True
        )
        fit_data, errors = utils.map_concurrent(
            garmin.parse_garmin_fit,
            {file.name.rsplit(".", 1)[0]: file for file in uploaded_fits},
            singleton=True,
        )
        print(fit_data)
        gpx_data.update(fit_data)

        if garmin_client:
            activities_tab, stats_tab = st.tabs(
                ["Load Activities", "Load Health Stats"])
            with activities_tab:
                garmin_data = garmin.garmin_activities_app(garmin_client)
                if garmin_data:
                    gpx_data.update(garmin_data)

            with stats_tab:
                garmin.garmin_stats_app(garmin_client)

    with strava_tab:
        strava_client = strava.connect_client()
        if strava_client:
            strava_data = strava.strava_app(strava_client)
            if strava_data:
                gpx_data.update(strava_data)

    gpx_data.update(data)

    if gpx_data:
        with st.expander("Plot activity Data"):
            plot_activity_data(gpx_data)

    analyse_gps_data(gpx_data)

    st.button("Logout", key='logout2', on_click=clear_state)


@st.fragment
def plot_activity_data(gps_data):
    tabs = st.tabs(gps_data)
    st.divider()
    with st.popover("Figure settings"):
        height = st.number_input(
            "Set profile figure height",
            100, None, 600, step=50,
            key='plot_activity_data_height',
        )

    for tab, (name, data) in zip(tabs, gps_data.items()):
        plot_data = data.reset_index()
        default = [c for c in DEFAULT_FACETS if c in data]
        options = default + list(plot_data.columns.difference(DEFAULT_FACETS))
        with tab:
            col0, col1, col2 = st.columns((1, 2, 2))
            with col0:
                smooth = st.number_input(
                    "Smooth data over",
                    0, None, value=0, step=5,
                    key=f'{name}_activity_data_smooth',
                )
            if smooth:
                data = plot_data.set_index("timeElapsed")
                number_data = data.select_dtypes('number', "timedelta").rolling(
                    f"{smooth}s"
                ).mean().reset_index()
                dt_data = data.select_dtypes('timedelta').apply(
                    lambda s: s.dt.total_seconds()
                ).rolling(
                    f"{smooth}s"
                ).mean().apply(
                    pd.to_timedelta, unit='s'
                ).reset_index()
                plot_data = pd.concat([
                    number_data, dt_data, plot_data.select_dtypes(["datetime"])
                ], axis=1)
            with col1:
                left_axis = st.multiselect(
                    "Plot on left axis",
                    key=f"{name}_plotleft",
                    # index=0,
                    default=default[:1],
                    options=options,
                )
            with col2:
                right_axis = st.multiselect(
                    "Plot on right axis",
                    key=f"{name}_plotright",
                    default=[],
                    options=options,
                )

            fig = go.Figure()
            for c in left_axis:
                fig = app.scatter(
                    plot_data, 'distance', c, fig=fig
                )
            for c2 in right_axis:
                fig = app.scatter(
                    plot_data, 'distance', c2, fig=fig, yaxis='y2'
                )

            fig.update_layout(
                height=height,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                yaxis=dict(
                    title="+".join(left_axis)
                ),
                yaxis2=dict(
                    title=" + ".join(right_axis)
                ),
                xaxis=dict(
                    title='Distance (km)'
                )
            )
            st.plotly_chart(fig, use_container_width=True)


def clear_state():
    st.query_params.clear()
    st.session_state.clear()
    st.cache_resource.clear()


@st.fragment
def analyse_gps_data(gpx_data):
    start = 0
    for k, data in gpx_data.items():
        data.index += start
        start += len(data)

    with st.expander("Landmarks"):
        st.subheader("Landmarks")
        set_landmarks = app.set_landmarks(gps_data=gpx_data)
        locations = set_landmarks.set_index(["location", "landmark"])

    if not gpx_data:
        st.write("No data uploaded")
        st.stop()
        raise st.runtime.scriptrunner.StopException()

    # with st.expander("Show map"):
    #     app.draw_gps_data(gpx_data, locations)

    with st.spinner("Processing Crossing Times"):
        crossing_times = app.get_crossing_times(
            gpx_data, locations=locations)
        if crossing_times:
            all_crossing_times = pd.concat(crossing_times, names=['name'])
        else:
            all_crossing_times = pd.DataFrame([])

    with st.expander("Piece selecter", expanded=True):
        st.subheader("Piece selecter")
        piece_information = app.select_pieces(
            all_crossing_times)

        if piece_information is None:
            st.write("No valid pieces could be found")
            keep = []
        else:
            options = []
            for d in gpx_data.values():
                options = d.columns.union(options)

            keep = st.multiselect(
                "Select data to keep",
                options=options,
                default=options.intersection(DEFAULT_FACETS),
            )
            average_cols = ['time'] + keep
            piece_information['piece_data'].update(
                splits.get_pieces_interval_averages(
                    piece_information['piece_data']['Timestamp'],
                    {k: d[[c for c in average_cols if c in d]]
                        for k, d in gpx_data.items()},
                    time='time'
                )
            )

            app.show_piece_data(piece_information['piece_data'])

    if piece_information:
        with st.expander("Plot Piece Profile"):
            st.subheader("Plot Piece Profile")
            plot_piece_profiles(
                piece_information, gpx_data, [app.PACE_TIME_COL] + keep)

    with st.expander("Timings summary"):
        st.subheader("Timings summary")
        if crossing_times:
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

            fig.update_layout(
                height=height,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
            )
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
            show_times.reset_index(), hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "time": st.column_config.TimeColumn(
                    "Time", format="HH:mm:ss.S"
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
                show_crossings.reset_index(), hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date"),
                    "time": st.column_config.TimeColumn(
                        "Time", format="HH:mm:ss.S"
                    )
                },
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
            st.dataframe(upload_timings.reset_index(), hide_index=True)
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
                show_times.loc[order.index].reset_index(),
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
