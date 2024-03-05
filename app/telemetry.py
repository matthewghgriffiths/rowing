
from tracemalloc import start
import streamlit as st
import io
from functools import partial
from pathlib import Path
import os
import sys

import io
import zipfile

import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

try:
    from rowing import utils
    from rowing.analysis import geodesy, splits, app, telemetry
except ImportError:
    DIRPATH = Path(__file__).resolve().parent
    LIBPATH = str(DIRPATH.parent)
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    from rowing import utils
    from rowing.analysis import geodesy, splits, app, telemetry

logger = logging.getLogger("telemetry")
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
# )


def main(state=None):
    state = state or {}
    telemetry_data = state.pop("telemetry_data", {})
    st.session_state.update(state)
    state = st.session_state or state

    logger.info("telemetry")
    st.set_page_config(
        page_title="Peach Telemetry Analysis",
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    """
    # Peach Telemetry processing
    """
    with st.sidebar:
        st.subheader("QR Code")
        st.image(
            "https://chart.googleapis.com/chart"
            "?cht=qr&chl=https%3A%2F%2Frowing-telemetry.streamlit.app"
            "&chs=360x360&choe=UTF-8&chld=L|0"
        )
        default_height = st.number_input(
            "Default Figure Height",
            min_value=100,
            max_value=3000,
            value=600,
            step=50,
        )
        if st.button("Reset State"):
            st.session_state.clear()
            st.cache_resource.clear()

    with st.expander("Upload Telemetry Data"):
        use_names = st.checkbox("Use crew list", True)
        tabs = st.tabs([
            "Upload text", "Upload csv", "Upload xlsx", "Upload Zip"
        ])
        with tabs[0]:
            uploaded_files = st.file_uploader(
                "Upload All Data Export from PowerLine (tab separated)",
                accept_multiple_files=True
            )
            st.write("Text should should be formatted like below (no commas!)")
            st.code(
                r""" 
                =====	File Info
                Serial #	Session	Filename	Start Time	TZBIAS	Location	Summary	Comments
                0000	156	<DATAPATH>\row000123-0000123D.peach-data	Sun, 01 Jan 2023 00:00:00	3600			
                =====	GPS Info
                Lat	Lon	UTC	PeachTime
                00.0000000000	00.0000000000	01 Jan 2023 00:00:00 (UTC)	00000
                =====	Crew Info
                ...
                """, None)
            telemetry_data.update(
                app.parse_telemetry_text(
                    uploaded_files, use_names=use_names, sep='\t'
                )
            )
        with tabs[1]:
            uploaded_files = st.file_uploader(
                "Upload All Data Export from PowerLine (comma separated)",
                accept_multiple_files=True
            )
            st.write("Text should should be formatted like below")
            st.code(
                r""" 
                =====,File Info,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                Serial #,Session,Filename,Start Time,TZBIAS,Location,Summary,Comments,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                0000,000,<DATAPATH>\row000123-0000123D.peach-data,"Sun, 01 Jan 2023 00:00:00",0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                =====,GPS Info,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                Lat,Lon,UTC,PeachTime,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                00.00000000,00.00000000,Sun 01 Jan 2023 00:00:00 (UTC),00000,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                ...
                """, None)
            telemetry_data.update(
                app.parse_telemetry_text(
                    uploaded_files, use_names=use_names, sep=','
                )
            )
        with tabs[2]:
            uploaded_files = st.file_uploader(
                "Upload Data Export from PowerLine",
                accept_multiple_files=True
            )
            telemetry_data.update(
                app.parse_telemetry_excel(
                    uploaded_files, use_names=use_names
                )
            )
        with tabs[3]:
            uploaded_files = st.file_uploader(
                "Upload Zip of Data Exports",
                accept_multiple_files=True
            )
            telemetry_data.update(
                app.parse_telemetry_zip(uploaded_files)
            )
            # st.spinner("Processing Zip Files")
            #     for file in uploaded_files:
            #         with zipfile.ZipFile(file) as z:
            #             for f in z.filelist:
            #                 name, key = f.filename.removesuffix(
            #                     ".parquet").split("/")
            #                 data = pd.read_parquet(
            #                     z.open(f.filename)
            #                 )
            #                 telemetry_data.setdefault(name, {})[key] = data

        gps_data = {
            k: split_data['positions'] for k, split_data in telemetry_data.items()
        }

        if st.toggle("save as zip"):
            with st.spinner("Creating Zip File"):
                zipdata = app.telemetry_to_zipfile(telemetry_data)

            st.download_button(
                label=":inbox_tray: Download Telemetry_Data.zip",
                file_name="Telemetry_Data.zip",
                data=zipdata,
            )

    with st.expander("Landmarks"):
        set_landmarks = app.set_landmarks(gps_data=gps_data)
        locations = set_landmarks.set_index(["location", "landmark"])

    if not telemetry_data:
        st.write("No data uploaded")
        st.stop()
        raise st.runtime.scriptrunner.StopException()

    logger.info("Show map")
    with st.expander("Show map"):
        fig = app.draw_gps_data(gps_data, locations)

    logger.info("Crossing Times")
    with st.spinner("Processing Crossing Times"):
        crossing_times = app.get_crossing_times(gps_data, locations=locations)
        all_crossing_times = pd.concat(crossing_times, names=['name'])

    if all_crossing_times.empty:
        return

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

        landmark_times = all_crossing_times.droplevel(
            ["location", "distance"]
        ).unstack("landmark")
        landmark_times = landmark_times.loc[
            landmark_times.min(1).sort_values().index,
            landmark_times.min().sort_values().index
        ]
        st.write(landmark_times)

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

    logger.info("Select piece start end")
    with st.expander("Select Piece start/end"):
        piece_data, start_landmark, finish_landmark, intervals = app.select_pieces(
            all_crossing_times
        )
        if piece_data is None:
            st.write("No valid pieces could be found")
        else:
            piece_data.update(
                telemetry.get_interval_averages(
                    piece_data['Timestamp'], telemetry_data)
            )
            app.show_piece_data(piece_data)

    logger.info("Plot piece data")
    telemetry_figures = {}
    with st.expander("Plot piece data", True):
        cols = st.columns((1, 1, 9))
        with cols[0]:
            all_plots = st.toggle(
                'Make all plots',
                value=state.get('Make all plots'),
                key='Make all plots')

        if piece_data:
            show_rowers = None
            with cols[1]:
                toggle_athletes = st.toggle(
                    "Filter athletes", key=f"toggleother"
                )
                with cols[2]:
                    if toggle_athletes:
                        cols2 = st.columns((3, 2, 2))
                    else:
                        cols2 = st.columns(2)

            with cols2[-2]:
                window = st.number_input(
                    "Select window to average over (s), set to 0 to remove smoothing",
                    value=10,
                    min_value=0,
                    step=5,
                )
            with cols2[-1]:
                height = st.number_input(
                    "Set figures height",
                    100, 3000, default_height, step=50,
                )

            piece_distances = piece_data['Total Distance']
            landmark_distances = piece_data['Distance Travelled'].mean()[
                piece_distances.columns
            ].sort_values()
            compare_power = telemetry.compare_piece_telemetry(
                telemetry_data, piece_data, gps_data, landmark_distances, window=int(window))
            n_legs = compare_power.groupby(
                ["name", "leg"]
            ).size().groupby(level=0).size()

            piece_data_filter = piece_data
            if toggle_athletes:
                with cols2[0]:
                    piece_rowers = compare_power.groupby(
                        ["Position", "name"]).size().index
                    show_rowers = st.multiselect(
                        "Select athletes to plot",
                        options=piece_rowers.map("|".join),
                        key="show_athletes",
                    )
                    if show_rowers:
                        show_rowers = pd.MultiIndex.from_tuples([
                            tuple(r.split("|", 2)) for r in show_rowers
                        ], names=["Position", "name"])
                        filter_rows = pd.MultiIndex.from_frame(
                            compare_power[["Position", "name"]]
                        ).isin(show_rowers)
                        compare_power = compare_power[filter_rows]
                        piece_data_filter = {
                            k: data.reindex(show_rowers.swaplevel(0, 1))
                            if data.index.nlevels == 2 and len(data.index.intersection(show_rowers.swaplevel(0, 1)))
                            else data
                            for k, data in piece_data.items()
                        }

            telemetry_figures = {}
            tab_names = ["Pace Boat"] + list(telemetry.FIELDS)
            telem_tabs = dict(zip(tab_names, st.tabs(tab_names)))
            for col, tab in tqdm(telem_tabs.items()):
                with tab:
                    cols = st.columns((1, 7))

                    with cols[0]:
                        on = st.toggle('Make plot', value=all_plots,
                                       key=col + ' make plot')

                    _height = height
                    facet_col_wrap = 4
                    if col == "Pace Boat" and on:
                        fig, time_behind = app.plot_pace_boat(
                            piece_data,
                            landmark_distances,
                            gps_data,
                            height=_height,
                            col=cols[1],
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.subheader("Time behind pace boat")
                        st.dataframe(time_behind)
                        telemetry_figures[
                            col, f"{start_landmark} to {finish_landmark}"] = fig

                    elif on:
                        if col == 'Work PC':
                            with cols[1]:
                                cols2 = st.columns(2)

                            n_plots = len(
                                compare_power[['name', 'leg', 'Position']].value_counts())
                            if show_rowers:
                                n_plots = len(show_rowers)

                            with cols2[0]:
                                facet_col_wrap = st.number_input(
                                    "Select number of columns",
                                    value=4, min_value=1, step=1,
                                )
                                n_rows = np.ceil(n_plots / facet_col_wrap)
                            with cols2[1]:
                                _height = st.number_input(
                                    "Set Work PC figure height",
                                    min_value=100,
                                    # 3000,
                                    value=int(height * n_rows // 2),
                                    step=50,
                                )

                        fig = app.make_telemetry_distance_figure(
                            compare_power, landmark_distances, col,
                            facet_col_wrap=facet_col_wrap,
                        )

                        itemclick = 'toggle'
                        itemdoubleclick = "toggleothers"
                        groupclick = 'toggleitem'
                        fig.update_layout(
                            title=f"{col}: {start_landmark} to {finish_landmark}",
                            height=_height,
                            legend=dict(
                                itemclick=itemclick,
                                itemdoubleclick=itemdoubleclick,
                                groupclick=groupclick,
                            )
                            # xaxis_title="Distance (km)",
                            # yaxis_title=col,
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        telemetry_figures[
                            col, f"{start_landmark} to {finish_landmark}"] = fig

                        st.write(
                            "Click on legend to toggle traces, "
                            "double click to select only one piece"
                        )

                    cols = st.columns(2)
                    with cols[0]:
                        interval_stats = piece_data_filter.get(
                            f"Interval {col}")
                        if interval_stats is not None:
                            st.subheader("Interval Averages")
                            st.write(interval_stats)

                    with cols[1]:
                        interval_stats = piece_data_filter.get(
                            f"Average {col}")
                        if interval_stats is not None:
                            st.subheader("Piece Averages")
                            st.write(interval_stats)

    with st.expander("Plot Stroke Profiles", True):
        if st.toggle(
            'Make profile plots',
            value=state.get('Make profile plots'),
            key='Make profile plots'
        ) and piece_data:
            profiles, boat_profiles, crew_profiles = app.make_stroke_profiles(
                telemetry_data, piece_data
            )

            crew_profile = pd.concat(
                crew_profiles, names=['name', 'leg']
            ).reset_index(['name', 'leg'])
            crew_profile['Rower'] = (
                # crew_profile.Position + "|" + crew_profile.File
                crew_profile.name + "|" + crew_profile.Position
            )
            n_legs = crew_profile.groupby(
                ["name", "leg"]
            ).size().groupby(level=0).size()

            tabs = st.tabs(
                ["Rower Profiles", "Boat Profile", "Grouped Profiles"])
            with tabs[0]:
                cols = st.columns(4)
                with cols[0]:
                    x = st.selectbox(
                        "Select x-axis",
                        ['GateAngle', 'Normalized Time', 'GateForceX',
                            'GateAngleVel', "GateAngle0"]
                    )
                with cols[1]:
                    y = st.selectbox(
                        "Select y-axis",
                        ['GateForceX', 'GateAngle', 'GateAngleVel',
                            "GateAngle0", 'Normalized Time']
                    )
                with cols[2]:
                    height = st.number_input(
                        "Set figure height",
                        key='rower profile fig height',
                        min_value=100,
                        max_value=None,
                        value=default_height,
                        step=100,
                    )
                with cols[3]:
                    ymin = float(min(
                        profile[y].min() for profile in crew_profiles.values()
                    ))
                    ymax = float(max(
                        profile[y].max() for profile in crew_profiles.values()
                    ))
                    yr = float(ymax - ymin)
                    yrange = st.slider(
                        "Set y lims",
                        ymin - yr/10, ymax + yr /
                        10, (ymin - yr/10, ymax + yr/10)
                    )

                for (name, leg), profile in crew_profiles.items():
                    fig = px.line(
                        profile,
                        x=x,
                        y=y,
                        color='Position',
                        title=name
                    )
                    fig.update_yaxes(
                        range=yrange
                    )
                    fig.update_layout(
                        height=height
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    telemetry_figures[f"{x}-{y}", name] = fig

            with tabs[2]:
                cols = st.columns(3)
                with cols[0]:
                    x = st.selectbox(
                        "Select x-axis",
                        ['GateAngle', 'Normalized Time', 'GateForceX',
                            'GateAngleVel', "GateAngle0"],
                        key="Select x-axis2",
                    )
                with cols[1]:
                    y = st.selectbox(
                        "Select y-axis",
                        ['GateForceX', 'GateAngle', 'GateAngleVel',
                            "GateAngle0", 'Normalized Time'],
                        key="Select y-axis2",
                    )
                with cols[2]:
                    height = st.number_input(
                        "Set figure height",
                        key='rower profile fig height2',
                        min_value=100,
                        max_value=None,
                        value=default_height,
                        step=100,
                    )

                fig = go.Figure()
                for (file, leg), piece_profile in crew_profile.groupby(["name", "leg"]):
                    for pos, profile in piece_profile.groupby("Position"):
                        name = file if n_legs[file] == 1 else f"{file} {leg=}"
                        fig.add_trace(
                            go.Scatter(
                                x=profile[x],
                                y=profile[y],
                                legendgroup=f"{name} {leg}",
                                legendgrouptitle_text=name,
                                name=pos,
                                mode='lines',
                            )
                        )

                fig.update_layout(
                    title=f"{start_landmark} to {finish_landmark}: {y} vs {x}",
                    height=height,
                    legend=dict(
                        itemclick='toggle',
                        itemdoubleclick='toggleothers',
                        groupclick='toggleitem',
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                telemetry_figures['profile', f'{x}-{y}'] = fig

            with tabs[1]:
                cols = st.columns(2)
                with cols[0]:
                    facets = st.multiselect(
                        "Select facets",
                        [
                            'Speed', 'Accel', 'Roll Angle', 'Pitch Angle', 'Yaw Angle'
                        ],
                        default=[
                            'Speed', 'Accel', 'Roll Angle', 'Pitch Angle', 'Yaw Angle'
                        ],
                    )
                with cols[1]:
                    height = st.number_input(
                        "Set figure height",
                        min_value=100,
                        max_value=None,
                        value=len(facets) * 200,
                        step=100,
                    )

                boat_profile = pd.concat(
                    boat_profiles, names=['name', 'leg']
                ).reset_index(["name", 'leg']).rename_axis(
                    columns='Measurement'
                ).set_index(
                    ["Normalized Time", "name", 'leg']
                )[facets].stack().rename("value").reset_index()
                fig = px.line(
                    boat_profile,
                    x="Normalized Time",
                    y="value",
                    color='name',
                    line_dash='leg',
                    facet_row='Measurement',
                    # title=name
                )
                fig.update_yaxes(matches=None, showticklabels=True)
                fig.update_layout(height=height)
                st.plotly_chart(fig, use_container_width=True)
                telemetry_figures['profile', 'boats'] = fig

    logger.info("Download data")
    with st.expander("Download Data"):
        cols = st.columns(4)
        if piece_data:
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

                        save_data.to_excel(xlf, name.replace("/", " per "))

                xldata.seek(0)
                st.download_button(
                    f":inbox_tray: Download telemetry-{start_landmark}-{finish_landmark}.xlsx",
                    xldata,
                    # type='primary',
                    file_name=f'telemetry-{start_landmark}-{finish_landmark}.xlsx'
                    # file_name="telemetry_piece_data.xlsx",
                )

        if telemetry_figures:
            save_figures = {
                f"{col}/{name}": fig for (col, name), fig in telemetry_figures.items()}
            with cols[2]:
                width = st.number_input(
                    "Save figure width",
                    value=1000,
                    min_value=50,
                    max_value=3000,
                    step=50,
                )
            with cols[3]:
                height = st.number_input(
                    "Save figure height",
                    value=default_height,
                    min_value=50,
                    max_value=3000,
                    step=50,
                )
            with cols[1]:
                download_figures = st.multiselect(
                    "Download Figures as",
                    options=['html', 'png', 'svg', 'pdf',
                             'jpg', 'webp', 'online-html'],
                    default=[],
                )
                for file_type in download_figures:
                    pio.templates.default = "plotly"
                    with st.spinner(f"Creating {file_type}s"):
                        file_name = f"telemetry-{start_landmark}-{finish_landmark}.{file_type}.zip"
                        kwargs = dict(
                            height=height,
                            width=width,
                        )
                        if file_type == 'html':
                            kwargs = {}
                        if file_type == 'online-html':
                            file_type = 'html'
                            kwargs = {"include_plotlyjs": "cdn"}

                        zipdata = app.figures_to_zipfile(
                            save_figures, file_type, **kwargs
                        )
                        st.download_button(
                            f":inbox_tray: Download {file_name}",
                            zipdata,
                            file_name=file_name,
                        )


if __name__ == "__main__":
    main()
