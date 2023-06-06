import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px

from rowing.world_rowing import api, livetracker, utils
from rowing.app import select, inputs, state, plots


st.set_page_config(
    page_title="World Rowing livetracker",
    layout='wide'
    # page_icon="👋",
)
st.title("World Rowing livetracker")

st.subheader("Select livetracker data")

races = select.select_races(filters=True).reset_index(drop=True)
boat_classes = races['boatClass.DisplayName'].unique()

with st.expander("Select GMTs"):
    st.text("Select competition best times")
    cbts = select.select_best_times(boat_classes)

    if cbts.empty:
        st.write("No GMTS selected")
        st.stop()

    st.text("Set GMT")
    gmts = select.set_gmts(cbts)

races = races.set_index("race.id").join(
    gmts.dt.total_seconds().rename("GMT"), on="boatClass.DisplayName"
)

with st.sidebar:
    threads = st.number_input(
        "number of threads to use", min_value=1, max_value=20, value=10, 
        step=1
    )
    threads = int(threads)
    download = st.checkbox("load livetracker data", len(races) < 30)

if not download:
    st.caption(f"Selected {len(races)} races")
    st.caption("Check load livetracker data in sidebar to view race data")
    st.stop()

live_data, intermediates = select.get_races_livedata(races, max_workers=threads)

with st.expander("Filter livetracker data"):
    live_data = select.filter_livetracker(live_data)

st.subheader("Show livetracker")

live_data, PGMT = select.set_livetracker_PGMT(live_data)

distance_from_pace = "distance from PGMT"
facets = ["PGMT", "distance from PGMT", "split", "strokeRate"]
fig = plots.make_livetracker_plot(
    facets, *plots.melt_livetracker_data(live_data, 100), 

)
st.plotly_chart(fig, use_container_width=True)