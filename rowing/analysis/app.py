import streamlit as st
import io 
import zipfile
import logging

import numpy as np
import pandas as pd 

import plotly.graph_objects as go
import plotly.express as px

from rowing.analysis import splits, files, geodesy, telemetry
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
def parse_telemetry_text(uploaded_files, use_names=True, sep='\t'):
    uploaded_data = {
        file.name.rsplit(".", 1)[0]: file.read().decode("utf-8")
        for file in uploaded_files
    }
    data, errs = utils.map_concurrent(
        telemetry.parse_powerline_text_data, 
        uploaded_data, 
        singleton=True, 
        use_names=use_names,
        sep=sep
    )
    if errs:
        logging.error(errs)
    
    return data


@st.cache_data
def parse_excel(file, use_names=True):
    data = pd.read_excel(file, header=None)
    return telemetry.parse_powerline_excel(data, use_names=use_names)


@st.cache_data
def parse_telemetry_excel(uploaded_files, use_names=True):
    uploaded_data = {
        file.name.rsplit(".", 1)[0]: file.read().decode("utf-8")
        for file in uploaded_files
    }
    data, errs = utils.map_concurrent(
        parse_excel, 
        uploaded_data, 
        singleton=True, 
        use_names=use_names,
    )
    if errs:
        logging.error(errs)
    
    return data


@st.cache_data
def get_crossing_times(gpx_data, locations=None, thresh=0.5):
    crossing_times, errors = utils.map_concurrent(
        splits.find_all_crossing_times, 
        gpx_data, 
        singleton=True, 
        locations=locations,
        thresh=thresh, 
    )
    if errors:
        logging.error(errors)
    return crossing_times

@st.cache_data
def get_location_timings(gpx_data, locations=None, thresh=0.5):
    location_timings, errors = utils.map_concurrent(
        splits.get_location_timings, gpx_data, 
        singleton=True, 
        locations=locations, 
        thresh=thresh, 
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
    landmark_distance = sel_times.reset_index(
        'distance'
    ).distance
    landmarks = landmark_distance.groupby(
        level=3).mean().sort_values().index
    landmark_dist = landmark_distance.unstack(
        'landmark').dropna(axis=1).mean(0)
    start, end = map(
        int, landmarks.get_indexer(
            [landmark_dist.idxmin(), landmark_dist.idxmax()]
        )
    )
    with cols[1]:
        start_landmark = st.selectbox(
            "select start landmark", 
            landmarks, 
            index=start, 
        )
    with cols[2]:
        finish_landmark = st.selectbox(
            "select finish landmark", 
            landmarks, 
            index=end, 
        )

    piece_data = splits.get_piece_times(sel_times, start_landmark, finish_landmark)
    return piece_data, start_landmark, finish_landmark

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
                "Set figure height", 100, 3000, 600, step=50, 
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
    data = locations
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


@st.cache_data
def make_stroke_profiles(telemetry_data, piece_data, nres=101):
    profiles = {}
    boat_profiles = {}
    crew_profiles = {}
    for piece, piece_times in piece_data['Timestamp'].iterrows():
        name = piece[1]
        profile = telemetry_data[name]['Periodic']
        start_time = piece_times.min()
        finish_time = piece_times.max()
        piece_profile = profile[
            profile.Time.dt.tz_localize(None).between(start_time, finish_time)
        ].set_index('Time').dropna(axis=1, how='all')

        profiles[name] = profile = telemetry.norm_stroke_profile(
            piece_profile, nres)

        mean_profile = profile.groupby(
            level=1
        ).mean().reset_index().rename(
            {"": "Boat"}, axis=1, level=1
        )
        boat_profiles[name] = boat_profile = mean_profile.xs("Boat", axis=1, level=1)
        
        crew_profiles[name] = mean_profile[
            mean_profile.columns.levels[0].difference(
                boat_profile.columns)
        ].rename_axis(
            columns=("Measurement", "Position")
        ).stack(1).reset_index("Position")

    return profiles, boat_profiles, crew_profiles



@st.cache_data
def make_telemetry_figures(telemetry_data, piece_data, window:int=0, tab_names=None):
    if tab_names is None:
        tab_names = [
            'Angle 0.7 F', 
            'Angle Max F', 
            'Average Power', 
            'AvgBoatSpeed',
            'CatchSlip', 
            'Dist/Stroke', 
            'Drive Start T', 
            'Drive Time',
            'FinishSlip', 
            'Max Force PC', 
            'MaxAngle', 
            'MinAngle', 
            'Length', 
            'Effective', 
            'Rating',
            'Recovery Time', 
            'Rower Swivel Power', 
            'StrokeNumber', 
            'SwivelPower',
            # 'Time', 
            'Work PC Q1', 
            'Work PC Q2',
            'Work PC Q3', 
            'Work PC Q4'
        ]

    telemetry_figures = {}
    for piece, piece_times in piece_data['Timestamp'].iterrows():
        name = piece[1]
        power = telemetry_data[name]['power']
        if window:
            time_power = power.set_index("Time").sort_index()
            avg_power = time_power.rolling(
                pd.Timedelta(seconds=window)
            ).mean()
            power = avg_power.reset_index()

        # piece_times = piece_data['Timestamp'].xs(name, level=1).iloc[0]
        start_time = piece_times.min()
        finish_time = piece_times.max()
        piece_power = power[
            power.Time.between(start_time, finish_time)
        ]
        piece_power.columns.names = 'Measurement', 'Position'

        epoch_times = (
            (piece_times - start_time) #+ pd.Timestamp(0)
        ).dt.total_seconds()
        for col in tab_names:
            plot_data = piece_power.stack(1)[
                ['Time', col]
            ]
            plot_data['Time'] = plot_data['Time'].ffill()
            plot_data['Elapsed'] = (
                (plot_data['Time'] - start_time) + pd.Timestamp(0)
            ).dt.tz_localize(None)
            plot_data = plot_data.dropna().reset_index()

            fig = px.line(
                plot_data, 
                x='Elapsed', 
                y=col, 
                color='Position',
                title=name, 
                template="streamlit",
                color_discrete_sequence=[
                    "#0068c9",
                    "#83c9ff",
                    "#ff2b2b",
                    "#ffabab",
                    "#29b09d",
                    "#7defa1",
                    "#ff8700",
                    "#ffd16a",
                    "#6d3fc0",
                    "#d5dae5",
                ],
            )
            for landmark, epoch in epoch_times.items():
                fig.add_vline(
                    x=int((epoch - 3600) * 1000), 
                    annotation_text=landmark, 
                    annotation=dict(
                        textangle=-90
                    )
                )
            fig.update_xaxes(
                tickformat="%M:%S",
                dtick=60*1000, 
                showgrid=True, 
                griddash='solid', 
            )
            fig.update_traces(visible=True)
            telemetry_figures[col, name] = fig

    return telemetry_figures

@st.cache_data
def figures_to_zipfile(figures, file_type, **kwargs):
    zipdata = io.BytesIO()
    with zipfile.ZipFile(zipdata, 'w') as zipf:
        for name, fig in figures.items():
            if file_type == 'html':
                fig_data = fig.to_html(**kwargs)
            else:
                fig_data = fig.to_image(format=file_type, **kwargs)

            zipf.writestr(
                f"{name}.{file_type}", fig_data
            )

    zipdata.seek(0)
    return zipdata