
import streamlit as st
import io
from pathlib import Path
import os
import sys

import logging

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

try:
    from rowing import utils
    from rowing.analysis import geodesy, splits, app
except ImportError:
    DIRPATH = Path(__file__).resolve().parent
    LIBPATH = str(DIRPATH.parent)
    realpaths = [os.path.realpath(p) for p in sys.path]
    if LIBPATH not in realpaths:
        sys.path.append(LIBPATH)

    from rowing import utils
    from rowing.analysis import geodesy, splits, app


logger = logging.getLogger(__name__)


def main(state=None):
    state = state or {}
    data = state.pop("gpx_data", {})
    st.session_state.update(state)
    state = st.session_state or state

    st.set_page_config(
        page_title="Rowing GPX",
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    """
    # GPX data processing
    """
    with st.sidebar:
        st.subheader("QR Code")
        st.image(
            "https://chart.googleapis.com/chart"
            "?cht=qr&chl=https%3A%2F%2Frowing-gps.streamlit.app"
            "&chs=360x360&choe=UTF-8&chld=L|0"
        )
        if st.button("Reset State"):
            st.session_state.clear()
            st.cache_resource.clear()

    uploaded_files = st.file_uploader(
        "Upload GPX files", accept_multiple_files=True
    )

    gpx_data, errors = utils.map_concurrent(
        app.parse_gpx,
        {file.name.rsplit(".", 1)[0]: file for file in uploaded_files},
        singleton=True,
    )
    gpx_data.update(data)

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
        all_crossing_times = pd.concat(crossing_times, names=['file'])
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
        piece_data, start_landmark, finish_landmark, intervals = app.select_pieces(
            all_crossing_times)
        if piece_data is None:
            st.write("No valid pieces could be found")
        else:
            app.show_piece_data(piece_data)

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
                ).applymap(
                    utils.format_timedelta, hours=True
                ).replace("00:00:00.00", "").T
                st.dataframe(upload_timings)
                app.download_csv(
                    f"{name}-timings.csv",
                    upload_timings,
                )

    with st.expander("Compare Piece Profile"):
        if piece_data:
            piece_distances = piece_data['Total Distance']
            piece_timestamps = piece_data['Timestamp']
            landmark_distances = piece_data['Distance Travelled'].mean()[
                piece_distances.columns]
            dists = np.arange(0, landmark_distances.max(), 0.005)

            piece_gps_data = {}
            for piece in piece_distances.index:
                positions = gpx_data[piece[1]].copy()
                positions.time = positions.time.dt.tz_localize(None)
                piece_gps_data[piece] = splits.get_piece_gps_data(
                    positions,
                    piece_distances.loc[piece],
                    piece_timestamps.loc[piece].dt.tz_localize(None),
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
                color='file',
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
                upload_timings = timings.droplevel(
                    "location"
                ).droplevel("location", axis=1).rename_axis(
                    ["", 'leg', 'landmark', 'distance']
                ).rename_axis(
                    ["", 'leg', 'landmark', 'distance'], axis=1
                ).applymap(
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


if __name__ == "__main__":
    main()
