import streamlit as st
import io 
import logging

import numpy as np
import pandas as pd 

import plotly.graph_objects as go

from rowing.analysis import splits, files, geodesy
from rowing import utils
from rowing.app import threads 


@st.cache_data
def parse_gpx(file):
    return files.parse_gpx_data(files.gpxpy.parse(file))

def download_csv(
        file_name, df, label=":inbox_tray: Download data as csv", csv_kws=None, **kwargs
    ):
    st.download_button(
        label=label, 
        file_name=file_name,
        data=df.to_csv(**(csv_kws or {})).encode("utf-8"), 
        mime="text/csv",
        **kwargs, 
    )

@st.cache_data
def get_crossing_times(gpx_data, locations=None):
    crossing_times, errors = utils.map_concurrent(
        splits.find_all_crossing_times, gpx_data, singleton=True, locations=locations
    )
    if errors:
        logging.error(errors)
    return crossing_times

@st.cache_data
def get_location_timings(gpx_data, locations=None):
    location_timings, errors = utils.map_concurrent(
        splits.get_location_timings, gpx_data, singleton=True, locations=locations
    )
    if errors:
        logging.error(errors)
    return location_timings

@st.cache_data
def get_fastest_times(gpx_data):
    best_times, errors = utils.map_concurrent(
        splits.find_all_best_times, 
        gpx_data, singleton=True, 
    )
    if errors:
        logging.error(errors)
    return best_times

def select_pieces(all_crossing_times):
    piece_dates = np.sort(all_crossing_times.dt.date.unique())
    cols = st.columns(3)
    with cols[0]:
        select_dates = st.multiselect(
            "select piece dates", 
            piece_dates,
            piece_dates,
        )
        select_dates = pd.to_datetime(select_dates).date

    
    sel_times = all_crossing_times[
        all_crossing_times.dt.date.isin(select_dates)
    ].sort_index(level=(0, 4))

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

    return splits.get_piece_times(sel_times, start_landmark, finish_landmark)

def show_piece_data(piece_data):
    tabs = st.tabs(list(piece_data))
    for tab, (key, data) in zip(tabs, piece_data.items()):
        with tab:
            data = data.copy()
            for c, col in data.items():
                if pd.api.types.is_timedelta64_dtype(col.dtype):
                    data[c] = col.map(utils.format_timedelta)

            st.dataframe(data)


def set_landmarks(landmarks=None, title=True):
    tab1, tab2 = st.tabs([
        "Edit Landmarks", "Upload Landmarks", #"Map of Landmarks"
    ])
    if landmarks is None:
        landmarks = splits.load_location_landmarks().reset_index()
    
    if title:
        landmarks['landmark'] = landmarks['landmark'].str.replace(
            "_", " "
        ).str.title().str.replace(
            r"([0-9][A-Z])", 
            lambda m: m.group(0).lower(), 
            regex=True
        )

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
            ).drop_duplicates().reset_index(drop=True)

        input = st.text_area(
            "Paste comma separated values in here, e.g. "
            "`tideway,boat_race_finish,51.4719098,-0.269101,122`"
        )
        if input:
            st.write("Entered Landmarks")
            new_landmarks = pd.read_csv(
                io.StringIO(input), 
                header=None, 
                names=landmarks.columns 
            )
            st.dataframe(new_landmarks)
            landmarks = pd.concat(
                [new_landmarks, landmarks]
            ).drop_duplicates().reset_index(drop=True)



    with tab1:
        col1, col2 = st.columns(2)
        locations = landmarks.location.unique()
        with col1:
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
        with col2:
            st.write(
                """
                Add new landmarks by entering at the bottom. 
                        
                Delete old landmarks by selecting the left column and pressing delete
                
                Hold shift to select multiple landmarks
                
                The current set of landmarks can be downloaded as a csv 
                
                A custom landmarks can be uploaded as a csv which will be merged with the existing landmarks. 
                This csv must match the format of the downloaded csv
                """)
        
        download_csv(
            "landmarks.csv", 
            set_landmarks, 
            ':inbox_tray: download set landmarks as csv', 
            csv_kws=dict(index=False), 
        )
        
    with tab2:
        download_csv(
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
            arrow = geodesy.make_arrow_base(landmark, 0.25, 0.1, 20)

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

    return set_landmarks


def draw_gps_data(gps_data, locations):
    cols = st.columns([5, 2])
    with cols[0]:
        map_style = st.selectbox(
            "map style", 
            ["open-street-map", "carto-positron", "carto-darkmatter"]
        )
    with cols[1]:
        height = st.number_input("Set figure height", 100, 2000, 600)

    fig = go.Figure()
    for name, data in gps_data.items():
        fig.add_trace(go.Scattermapbox(
            lon = data.longitude, 
            lat = data.latitude,
            mode = 'lines',
            name = name, 
        ))
    fig.add_trace(go.Scattermapbox(
        lon = locations.longitude, 
        lat = locations.latitude,
        # hoverinfo = landmark_locs.index,
        mode = 'markers+text',
        name = 'Landmarks',
        text = locations.index.get_level_values("landmark"), 
        marker={
            'size': 20, 
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