
import streamlit as st
import io
from functools import partial
from pathlib import Path
import os
import sys
import yaml
import json

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
                accept_multiple_files=True,
                type=['xlsx', 'xls'],
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

    logger.info("Show heat map")
    with st.expander("Show heat map"):
        cols = st.columns((4, 4, 2))
        with cols[-1]:
            with st.popover("heat map settings") as heat_map_settings:
                dists = st.number_input(
                    "marker spacing (m)", min_value=1, value=5
                ) / 1000
                height = st.number_input(
                    "Heatmap height (px)", min_value=100, step=50, value=default_height
                )
                marker_size = st.number_input(
                    "Marker size", min_value=1, value=10
                )
                map_style = st.selectbox(
                    "map style",
                    ["open-street-map", "carto-positron", "carto-darkmatter"],
                    key='heatmap_style'
                )
                colorscales = px.colors.named_colorscales()
                colorscale = st.selectbox(
                    "heatmap color scale",
                    colorscales,
                    index=colorscales.index('plasma'),
                )
                cmin = st.number_input(
                    "Color scale min (clear for autoscaling)", value=None)
                cmid = st.number_input(
                    "Color scale mid (clear for autoscaling)", value=None)
                cmax = st.number_input(
                    "Color scale max (clear for autoscaling)", value=None)

        c0 = 'AvgBoatSpeed'
        c1 = 'Boat'
        file_col = {}
        for k, data in telemetry_data.items():
            data = data['power']
            with cols[0]:
                _options = [
                    'Angle 0.7 F', 'Angle Max F', 'Average Power', 'AvgBoatSpeed',
                    'CatchSlip', 'Dist/Stroke', 'Drive Start T', 'Drive Time',
                    'Effective', 'FinishSlip', 'Length', 'Max Force PC', 'MaxAngle',
                    'MinAngle', 'Rating', 'Recovery Time', 'Rower Swivel Power',
                    'StrokeNumber', 'SwivelPower''Work PC Q1', 'Work PC Q2',
                    'Work PC Q3', 'Work PC Q4'
                ]
                options = data.columns.levels[0].intersection(_options)
                index = (
                    int(options.get_indexer_for([c0])[0]) if c0 in options else 0)
                c0 = st.selectbox(
                    f"choose data type to plot for {k}",
                    options=options,
                    index=index
                )

            with cols[1]:
                options = data[c0].columns.get_level_values(0)
                index = (
                    int(options.get_indexer_for([c1])[0]) if c1 in options else 0)
                c1 = st.selectbox(
                    f"choose data to plot for {k}",
                    options=options,
                    index=index
                )

            file_col[k] = (c0, c1)

        fig = app.make_gps_heatmap(
            telemetry_data, dists, file_col, marker_size=marker_size, map_style=map_style, height=height)
        fig.update_coloraxes(
            cmin=cmin,
            cmid=cmid,
            cmax=cmax,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                outlinewidth=0,
            )
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    logger.info("Crossing Times")
    with st.spinner("Processing Crossing Times"):
        crossing_times = app.get_crossing_times(gps_data, locations=locations)
        all_crossing_times = pd.concat(crossing_times, names=['name'])

    if all_crossing_times.empty:
        return

    with st.expander("Crossing times"):
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

        st.subheader("By file")
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
        piece_information = app.select_pieces(all_crossing_times)
        if piece_information is None:
            st.write("No valid pieces could be found")
        else:
            piece_information['piece_data'].update(
                telemetry.get_interval_averages(
                    piece_information['piece_data']['Timestamp'],
                    telemetry_data)
            )
            piece_information['gps_data'] = gps_data
            piece_information['telemetry_data'] = telemetry_data
            piece_information.update(app.make_stroke_profiles(
                piece_information['telemetry_data'],
                piece_information['piece_data']
            ))
            piece_information['piece_rowers'] = pd.MultiIndex.from_tuples([
                (r, k)
                for k, data in telemetry_data.items()
                for r in data['power'].columns.levels[1]
                if r
            ], names=('Position', 'name'))

            app.show_piece_data(piece_information['piece_data'])

    logger.info("Plot piece data")
    telemetry_figures = {}
    with st.expander("Plot piece data", True):
        if piece_information:
            window, show_rowers, all_plots, height = app.setup_plots(
                piece_information['piece_rowers'], state, default_height=default_height)
            piece_information = app.setup_plot_data(
                piece_information, window, show_rowers)

            tab_names = ["Pace Boat"] + list(telemetry.FIELDS)
            telem_tabs = dict(zip(tab_names, st.tabs(tab_names)))
            for col, tab in telem_tabs.items():
                with tab:
                    cols = st.columns((1, 7))
                    with cols[0]:
                        on = st.toggle('Make plot', value=all_plots,
                                       key=col + ' make plot')

                    if on:
                        figures, tables = app.plot_piece_col(
                            col, piece_information,
                            default_height=default_height,
                            key=col, input_container=cols[1]
                        )
                        for c, fig in figures.items():
                            st.plotly_chart(fig, use_container_width=True)
                        for t, table in tables.items():
                            st.subheader(t)
                            st.dataframe(table, use_container_width=True)

    with st.expander("Plot Stroke Profiles", True):
        if st.toggle(
            'Make profile plots',
            key='Make profile plots'
        ) and piece_information:
            tabs = st.tabs(
                ["Rower Profiles", "Boat Profile", "Grouped Profiles"])
            with tabs[0]:
                figures, tables = app.plot_rower_profiles(
                    piece_information, default_height=default_height)

                for c, fig in figures.items():
                    st.plotly_chart(fig, use_container_width=True)
                for t, table in tables.items():
                    st.subheader(t)
                    st.dataframe(table, use_container_width=True)

            with tabs[1]:
                figures, tables = app.plot_boat_profile(
                    piece_information, default_height=default_height)
                for c, fig in figures.items():
                    st.plotly_chart(fig, use_container_width=True)
                for t, table in tables.items():
                    st.subheader(t)
                    st.dataframe(table, use_container_width=True)

            with tabs[2]:
                figures, tables = app.plot_crew_profile(
                    piece_information, default_height=default_height)
                for c, fig in figures.items():
                    st.plotly_chart(fig, use_container_width=True)
                for t, table in tables.items():
                    st.subheader(t)
                    st.dataframe(table, use_container_width=True)

    with st.expander("Report"):

        report = st.container()

        st.subheader("Report settings")

        template_col, *settings_cols = st.columns((1, 1, 1, 4))
        with template_col:
            template_container = st.popover("Report template")

        with template_container:
            if st.button("use default"):
                for k0, vs in app.DEFAULT_REPORT.items():
                    for k1, v in vs.items():
                        k = ".".join([k0, k1])
                        st.session_state[k] = v
            else:
                template = st.file_uploader(
                    "Upload report template",
                    type=['yaml', 'json'],
                )
                if template:
                    template_data = yaml.safe_load(template)
                    print(json.dumps(template_data, indent=4))

                    for k0, vs in template_data.items():
                        for k1, v in vs.items():
                            k = ".".join([k0, k1])
                            st.session_state[k] = v

        window, show_rowers, n_views, height = app.setup_plots(
            piece_information['piece_rowers'], state,
            key='report_setup.',
            cols=settings_cols,
            toggle=False, nview=True, default_height=default_height
        )
        piece_information = app.setup_plot_data(
            piece_information, window, show_rowers)

        with report:
            for i in range(n_views):
                key = f"report_{i}."

                pop, title = st.columns((1, 6))
                with pop:
                    inputs = st.popover("Select Panel")
                cols = [inputs] * 3
                # with st.popover("Select Panel"):
                #     cols = st.columns((1, 1, 4))

                figures = {}

                with cols[0]:
                    plot_type = st.selectbox(
                        "What would you like to plot?",
                        key=key + "select plot",
                        options=[
                            "Piece profile",
                            "Stroke profile",
                            "Interval averages",
                            "Piece averages"
                        ]
                    )

                if plot_type == "Piece profile":
                    with cols[1]:
                        plot_data_type = st.selectbox(
                            "What data would you like to plot?",
                            key=key + "select piece",
                            options=["Pace Boat"] + list(telemetry.FIELDS)
                        )
                    figures, tables = app.plot_piece_col(
                        plot_data_type, piece_information, default_height=height, key=key,
                        input_container=cols[-1]
                    )

                elif plot_type == "Stroke profile":
                    with cols[1]:
                        plot_data_type = st.selectbox(
                            "What data would you like to plot?",
                            key=key + "select stroke",
                            options=[
                                "Rower profile",
                                "Crew profiles",
                                "Boat profile",
                            ]
                        )

                    if plot_data_type == "Rower profile":
                        figures, tables = app.plot_rower_profiles(
                            piece_information, default_height=height, key=key,
                            cols=cols)
                    elif plot_data_type == "Crew profiles":
                        figures, tables = app.plot_crew_profile(
                            piece_information, default_height=height, key=key,
                            cols=cols)
                    elif plot_data_type == "Boat profile":
                        figures, tables = app.plot_boat_profile(
                            piece_information, default_height=height, key=key,
                            cols=cols)

                elif plot_type == "Interval averages":
                    with cols[1]:
                        plot_data_type = st.selectbox(
                            "What data would you like to show?",
                            key=key + "select piece",
                            options=list(telemetry.FIELDS)
                        )
                        tables = {
                            f"Interval {plot_data_type}": piece_information['piece_data_filter'].get(
                                f"Interval {plot_data_type}")
                        }

                elif plot_type == "Piece averages":
                    with cols[1]:
                        plot_data_type = st.selectbox(
                            "What data would you like to show?",
                            key=key + "select piece",
                            options=list(telemetry.FIELDS)
                        )
                        tables = {
                            f"Average {plot_data_type}": piece_information['piece_data_filter'].get(
                                f"Average {plot_data_type}")
                        }

                with title:
                    st.subheader(f"{plot_type}: {plot_data_type}")

                for c, fig in figures.items():
                    st.subheader(c)
                    st.plotly_chart(fig, use_container_width=True)
                for t, table in tables.items():
                    st.subheader(t)
                    st.dataframe(
                        table,
                        height=(len(table) + 1) * 35 + 3,
                        use_container_width=True
                    )

        with template_container:
            report_state = {}
            for k, v in st.session_state.items():
                if k.startswith("report"):
                    k0, k1 = k.split(".", maxsplit=1)
                    report_state.setdefault(k0, {})[k1] = v

            report_settings = yaml.safe_dump(report_state)
            st.download_button(
                ":inbox_tray: Download telemetry_report_template.yaml",
                report_settings,
                "telemetry_report_template.yaml",
            )

    logger.info("Download data")
    with st.expander("Download Data"):
        cols = st.columns(4)
        if piece_information:
            with cols[0], st.spinner("Generating excel file"):
                xldata = io.BytesIO()
                with pd.ExcelWriter(xldata) as xlf:
                    for name, data in piece_information['piece_data'].items():
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
                start_landmark = piece_information['start_landmark']
                finish_landmark = piece_information['finish_landmark']
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
