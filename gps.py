import streamlit as st
import io 

import logging 

import numpy as np
import pandas as pd 

import plotly.graph_objects as go

from rowing.analysis import geodesy, splits, app
from rowing import utils

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Rowing GPX",
    layout='wide'
)
"""
# GPX data processing
"""

uploaded_files = st.file_uploader(
    "Upload GPX files", accept_multiple_files=True
)

with st.expander("Landmarks"):
    tab1, tab2 = st.tabs([
        "Edit Landmarks", "Upload Landmarks", #"Map of Landmarks"
    ])
    landmarks = splits.load_location_landmarks().reset_index()
    with tab2:
        uploaded = st.file_uploader(
            "Upload landmarks csv", 
            accept_multiple_files=False
        )
        if uploaded:
            uploaded_landmarks = pd.read_csv(uploaded)
            st.write("Uploaded Landmarks")
            st.dataframe(uploaded_landmarks, hide_index=True)
            landmarks = pd.concat(
                [uploaded_landmarks, landmarks]
            ).drop_duplicates()


    with tab1:
        locations = landmarks.location.unique()
        sel_locations = st.multiselect(
            "filter", locations, default=locations
        )
        set_landmarks = st.data_editor(
            landmarks[
                landmarks.location.isin(sel_locations)
            ], 
            hide_index=True, 
            num_rows="dynamic"
        )
        app.download_csv(
            "landmarks.csv", 
            set_landmarks, 
            ':inbox_tray: download set landmarks as csv', 
            csv_kws=dict(index=False), 
        )
        
    with tab2:
        app.download_csv(
            "landmarks.csv", 
            set_landmarks, 
            ':inbox_tray: download landmarks as csv', 
            csv_kws=dict(index=False), 
        )

    # with tab3: 
    if True:
        st.subheader("Map of Landmarks")       
        cols = st.columns([5, 2])
        with cols[0]:
            map_style = st.selectbox(
                "map style", 
                ["open-street-map", "carto-positron", "carto-darkmatter"],
                key='landmark map style'

            )
        with cols[1]:
            height = st.number_input(
                "Set figure height", 100, 2000, 600,
                key='landmark map height'
            )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scattermapbox(
            lon = set_landmarks.longitude, 
            lat = set_landmarks.latitude,
            # customdata = set_landmarks,
            mode = 'markers+text',
            name = 'Landmarks',
            text = set_landmarks.landmark, 
            cluster=dict(
                enabled=True, 
                maxzoom=5, 
                step=1, 
                size=20, 
            ),
            marker={
                'size': 5, 
                # 'symbol': "airfield", 
                # 'icon': dict(iconUrl="https://api.iconify.design/maki-city-15.svg"),
            },
            textposition='bottom right',
        ))

        for i, landmark in set_landmarks.iterrows():
            arrow = geodesy.make_arrow_base(landmark, 0.3, 0.1, 20)

            fig.add_trace(go.Scattermapbox(
                lon = arrow.longitude, 
                lat = arrow.latitude,
                # hoverinfo = landmark_locs.index,
                mode = 'lines',
                name = landmark.landmark,
                fill = 'toself', 
                hovertext = f"bearing={landmark.bearing:.1f}",
                line = dict(
                    width=3, 
                ),
                # text = list(set_landmarks.landmark), 
                # marker={
                #     'size': 5, 
                #     # 'symbol': landmark_locs.index,
                # },
                textposition='bottom right',
            ))

        
        fig.update_layout(
            mapbox = {
                'style': map_style,
                'center': {
                    'lon': set_landmarks.longitude.mean(), 
                    'lat': set_landmarks.latitude.mean(), 
                },
                'zoom': 5
            },
            showlegend=False,
            height=height, 
        )
        st.plotly_chart(fig, use_container_width=True)

gpx_data, errors = utils.map_concurrent(
    app.parse_gpx, 
    {file.name.rsplit(".", 1)[0]: file for file in uploaded_files}, 
    singleton=True
)

if not gpx_data:
    st.write("No data uploaded")
    st.stop()

with st.expander("Show map"):
    cols = st.columns([5, 2])
    with cols[0]:
        map_style = st.selectbox(
            "map style", 
            ["open-street-map", "carto-positron", "carto-darkmatter"]
        )
    with cols[1]:
        height = st.number_input("Set figure height", 100, 2000, 600)


    fig = go.Figure()
    for name, data in gpx_data.items():
        fig.add_trace(go.Scattermapbox(
            lon = data.longitude, 
            lat = data.latitude,
            mode = 'lines',
            name = name, 
        ))
    landmarks = splits.load_landmarks()
    fig.add_trace(go.Scattermapbox(
        lon = landmarks.longitude, 
        lat = landmarks.latitude,
        # hoverinfo = landmark_locs.index,
        mode = 'markers+text',
        name = 'Landmarks',
        text = landmarks.index, 
        marker={
            'size': 10, 
            # 'symbol': "airfield", 
            # 'icon': dict(iconUrl="https://api.iconify.design/maki-city-15.svg"),
        },
        textposition='bottom right',
    ))

    fig.update_layout(
        mapbox = {
            'style': map_style,
            'center': {
                'lon': data.longitude.mean(), 
                'lat': data.latitude.mean(), 
            },
            'zoom': 10
        },
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-.02,
            xanchor="right",
            x=1
        ),
        height=height, 
    )
    st.plotly_chart(
        fig, use_container_width=True
    )

with st.spinner("Processing Crossing Times"):
    crossing_times = app.get_crossing_times(gpx_data)
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
    landmarks, ind = np.unique(
        sel_times.index.get_level_values(3), return_index=True)
    landmarks = landmarks[np.argsort(ind)]
    with cols[1]:
        start_landmark = st.selectbox(
            "select start landmark", 
            landmarks, 
            index=0, 
        )
    with cols[2]:
        finish_landmark = st.selectbox(
            "select finish landmark", 
            landmarks, 
            index=landmarks.size-1, 
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
            ["Start Time", "file", "leg", "location"]
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
    location_timings = app.get_location_timings(gpx_data)

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
            app.download_csv(
                f"{name}-timings.csv",
                upload_timings,
            )

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
