from rowing import utils
from rowing.analysis import geodesy, splits, app, telemetry
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
import plotly.io as pio

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent)
realpaths = [os.path.realpath(p) for p in sys.path]
if LIBPATH not in realpaths:
    sys.path.append(LIBPATH)


logger = logging.getLogger("telemetry")
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
# )


logger.info("telemetry")
st.set_page_config(
    page_title="Peach Telemetry Analysis",
    layout='wide'
)
"""
# Peach Telemetry processing
"""

with st.expander("Upload Telemetry Data"):
    use_names = st.checkbox("Use crew list", True)
    tabs = st.tabs([
        "Upload text", "Upload csv", "Upload xlsx",
    ])
    telemetry_data = {}
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

    gps_data = {
        k: split_data['positions'] for k, split_data in telemetry_data.items()
    }

with st.expander("Landmarks"):
    set_landmarks = app.set_landmarks()
    locations = set_landmarks.set_index(["location", "landmark"])

if not telemetry_data:
    st.write("No data uploaded")
    st.stop()

logger.info("Show map")
with st.expander("Show map"):
    app.draw_gps_data(gps_data, locations)

logger.info("Crossing Times")
with st.spinner("Processing Crossing Times"):
    crossing_times = app.get_crossing_times(gps_data, locations=locations)
    all_crossing_times = pd.concat(crossing_times, names=['name'])

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
    if crossing_times:
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
        piece_compare_gps = app.align_pieces(
            piece_data, start_landmark, finish_landmark, gps_data, 0.005)

        if intervals:
            sep = intervals/1e3
            piece_gps = app.align_pieces(
                piece_data, start_landmark, finish_landmark, gps_data, sep
            )
            interval_locations = pd.concat({
                v: piece_gps.xs(v, level=-1, axis=1).mean(1)
                for v in ["latitude", 'longitude', 'bearing']
            }, axis=1).iloc[1:]
            interval_locations.index = interval_locations.index.map(
                "{:.2f} km".format)
            start_finish = locations.loc[(
                slice(None), [start_landmark, finish_landmark]), :]
            new_locations = pd.concat([
                start_finish,
                # locations,
                pd.concat({"intervals": interval_locations})
            ]).rename_axis(index=locations.index.names)
            crossing_times = app.get_crossing_times(
                gps_data, locations=new_locations)
            all_crossing_times = pd.concat(crossing_times, names=['name'])
            sel_times = all_crossing_times.sort_index(
                level=(4,)).droplevel('location')
            piece_data = splits.get_piece_times(
                sel_times, start_landmark, finish_landmark
            )

        avg_telem, interval_telem = {}, {}
        for piece, timestamps in piece_data['Timestamp'].iterrows():
            name = piece[1]
            power = telemetry_data[name][
                'power'
            ].sort_values("Time").reset_index(drop=True)
            avgP, intervalP = splits.get_interval_averages(
                power.drop("Time", axis=1, level=0),
                power.Time,
                timestamps
            )
            for k in avgP.columns.remove_unused_levels().levels[0]:
                avg_telem.setdefault(k, {})[name] = avgP[k].T
            for k in intervalP.columns.remove_unused_levels().levels[0]:
                interval_telem.setdefault(k, {})[name] = intervalP[k].T

        for k, data in avg_telem.items():
            piece_data[f"Average {k}"] = pd.concat(
                data, names=['name', 'position'])
        for k, data in interval_telem.items():
            piece_data[f"Interval {k}"] = pd.concat(
                data, names=['name', 'position'])

        app.show_piece_data(piece_data)

logger.info("Plot piece data")
telemetry_figures = {}
with st.expander("Plot piece data", True):
    cols = st.columns(3)
    with cols[0]:
        all_plots = st.toggle('Make all plots')

    if piece_data:
        with cols[1]:
            window = st.number_input(
                "Select window to average over (s), set to 0 to remove smoothing",
                value=10,
                min_value=0,
                step=5,
            )
        with cols[2]:
            height = st.number_input(
                "Set figures height",
                100, 3000, 600, step=50,
            )

        telemetry_plot_data = {}
        for piece, piece_times in piece_data['Timestamp'].iterrows():
            name = piece[1]
            power = telemetry_data[name]['power']
            if window:
                time_power = power.set_index("Time").sort_index()
                avg_power = time_power.rolling(
                    pd.Timedelta(seconds=window)
                ).mean()
                power = avg_power.reset_index()

            start_time = piece_times.min()
            finish_time = piece_times.max()
            piece_power = power[
                power.Time.between(start_time, finish_time)
            ]
            piece_power.columns.names = 'Measurement', 'Position'
            epoch_times = (piece_times - start_time).dt.total_seconds()
            telemetry_plot_data[name] = (
                piece_power, name, start_time, epoch_times)

        telemetry_figures = {}
        telem_tabs = dict(zip(telemetry.FIELDS, st.tabs(telemetry.FIELDS)))
        for col, tab in telem_tabs.items():
            with tab:
                on = st.toggle('Make plot', value=all_plots,
                               key=col + ' make plot')
                if on:
                    for name in piece_data['Timestamp'].index.get_level_values(1):
                        piece_power, name, start_time, epoch_times = telemetry_plot_data[name]
                        fig = app.make_telemetry_figure(
                            piece_power, col, name, start_time, epoch_times)
                        fig.update_layout(
                            height=height,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        telemetry_figures[col, name] = fig

                cols = st.columns(2)
                with cols[0]:
                    st.subheader("Interval Averages")
                    interval_stats = piece_data.get(f"Interval {col}")
                    if interval_stats is not None:
                        st.write(interval_stats)

                with cols[1]:
                    st.subheader("Piece Averages")
                    interval_stats = piece_data.get(f"Average {col}")
                    if interval_stats is not None:
                        st.write(interval_stats)

        # telemetry_figures = app.make_telemetry_figures(
        #     telemetry_data, piece_data, window=window, tab_names=tab_names
        # )
        # tab_names = st.multiselect(
        #     "Select data fields to plot",
        #     options=telemetry.FIELDS,
        #     default=telemetry.FIELDS,
        # )
        # tab_names = []
        # if tab_names:
        #     telem_tabs = dict(zip(tab_names, st.tabs(tab_names)))
        #     for col, tab in telem_tabs.items():
        #         with tab:
        #             for name in piece_data['Timestamp'].index.get_level_values(1):
        #                 fig = telemetry_figures[col, name]
        #                 fig.update_layout(
        #                     height=height,
        #                 )
        #                 st.plotly_chart(fig, use_container_width=True)

        #             st.subheader("Interval Averages")
        #             interval_stats = piece_data.get(f"Interval {col}")
        #             if interval_stats is not None:
        #                 st.write(interval_stats)

        #             st.subheader("Piece Averages")
        #             interval_stats = piece_data.get(f"Average {col}")
        #             if interval_stats is not None:
        #                 st.write(interval_stats)

with st.expander("Plot Stroke Profiles", True):
    if st.toggle('Make profile plots'):
        profiles, boat_profiles, crew_profiles = app.make_stroke_profiles(
            telemetry_data, piece_data
        )

        tabs = st.tabs(["Rower Profiles", "Boat Profile", "Grouped Profiles"])
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
                    value=500,
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
                    ymin - yr/10, ymax + yr/10, (ymin - yr/10, ymax + yr/10)
                )

            for name, profile in crew_profiles.items():
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
                st.plotly_chart(fig)
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
                    value=500,
                    step=100,
                )

            crew_profile = pd.concat(
                crew_profiles, names=['name']
            ).reset_index("name")
            crew_profile['Rower'] = (
                # crew_profile.Position + "|" + crew_profile.File
                crew_profile.name + "|" + crew_profile.Position
            )
            fig = px.line(
                crew_profile,
                x=x,
                y=y,
                color='Rower',
            )
            fig.update_layout(
                height=height,
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
                boat_profiles, names=['name']
            ).reset_index("name").rename_axis(
                columns='Measurement'
            ).set_index(
                ["Normalized Time", "name"]
            )[facets].stack().rename("value").reset_index()
            fig = px.line(
                boat_profile,
                x="Normalized Time",
                y="value",
                color='name',
                facet_row='Measurement',
                # title=name
            )
            fig.update_yaxes(matches=None, showticklabels=True)
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)
            telemetry_figures['profile', 'boats'] = fig


with st.expander("Compare Piece Profile"):
    if piece_data:
        piece_distances = piece_data['Total Distance']
        piece_timestamps = piece_data['Timestamp']
        landmark_distances = piece_data['Distance Travelled'].mean()[
            piece_distances.columns]
        dists = np.arange(0, landmark_distances.max(), 0.005)

        piece_gps_data = {}
        for piece in piece_distances.index:
            positions = gps_data[piece[1]]
            piece_gps_data[piece] = splits.get_piece_gps_data(
                positions,
                piece_distances.loc[piece],
                piece_timestamps.loc[piece],
                start_landmark,
                finish_landmark,
                landmark_distances
            )

        piece_compare_gps = pd.concat({
            piece: sel_data.set_index(
                "Distance Travelled"
            ).apply(utils.interpolate_series, index=dists)
            for piece, sel_data in piece_gps_data.items()
        }, axis=1, names=piece_distances.index.names
        ).rename_axis(
            index='distance'
        )
        boat_times = piece_compare_gps.xs('timeElapsed', level=-1, axis=1)
        pace_boat_finish = boat_times.iloc[-1].rename(
            "Pace boat time") + pd.Timestamp(0)
        pace_boat_finish[:] = pace_boat_finish.min()
        cols = st.columns(2)
        with cols[0]:
            st.write("Set pace boat time")
            pace_boat_finish = st.data_editor(
                pace_boat_finish.reset_index(),
                disabled=boat_times.columns.names,
                column_config={
                    "Pace boat time": st.column_config.TimeColumn(
                        "Pace boat time",
                        format="m:ss.S",
                        step=1,
                    ),
                }
            )
        with cols[1]:
            height = st.number_input(
                "Set figure height",
                100, 3000, 600, step=50,
                key="height piece profile",
            )

        pace_boat_finish['Pace boat time'] -= pd.Timestamp(0)
        pace_boat_finish = pace_boat_finish.set_index(
            boat_times.columns.names)['Pace boat time']
        pace_boat_time = pd.DataFrame(
            pace_boat_finish.values[None, :] * dists[:, None] / dists[-1],
            index=boat_times.index, columns=pace_boat_finish.index
        )
        time_behind = (
            boat_times - pace_boat_time
        ).unstack().dt.total_seconds().rename("time behind pace boat (s)").reset_index()

        fig = px.line(
            time_behind,
            x='distance',
            y='time behind pace boat (s)',
            color='name',
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=height)
        telemetry_figures['pacing', 'time behind pace boat'] = fig
        st.plotly_chart(fig, use_container_width=True)


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
                ":inbox_tray: Download telemetry_piece_data.xlsx",
                xldata,
                # type='primary',
                file_name="telemetry_piece_data.xlsx",
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
                value=600,
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
