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

with st.sidebar:
    download = st.checkbox("automatically load livetracker data", True)
    with st.expander("Settings"):
        threads = st.number_input(
            "number of threads to use", min_value=1, max_value=20, 
            value=state.get("threads", 6), 
            step=1
        )
        threads = int(threads)
        state.set("threads", threads)

st.subheader("Select livetracker data")

races = select.select_races(
    filters=True, select_all=False
).reset_index(drop=True)
boat_classes = races['boatClass.DisplayName'].unique()

if races.empty:
    if state.get("expander.filter_races", False):
        print(state.STATE)
        st.caption("select races to load")
        st.stop()

    state.set("expander.filter_races", True)
    state.update_query_params()
    st.experimental_rerun()

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

# if 1:
#     race_ids = races.index 
#     load_livetracker = livetracker.load_livetracker
#     max_workers=10
#     kwargs = {}
#     race_livetracks, errors = utils.map_concurrent(
#         load_livetracker, 
#         {race_id: race_id for race_id in race_ids},
#         singleton=True, max_workers=max_workers, **kwargs
#     )
#     results, errors = utils.map_concurrent(
#         livetracker.estimate_livetracker_times, 
#         race_livetracks, max_workers=max_workers, **kwargs
#     )
#     for race_id, data in race_livetracks.items():
#         print(race_id)
#         print(data)
#         print(livetracker.estimate_livetracker_times(*data))
#     # print(race_livetracks)
#     st.stop()
#     intermediates = pd.concat(
#         {race_id: inters for race_id, (_, inters) in results.items()}, 
#         axis=1
#     )
#     races_live_data = pd.concat(
#         {race_id: live_data for race_id, (live_data, _) in results.items()}, 
#         axis=0
#     ).reset_index(drop=True)

# livetracker.get_races_livetracks(
#     races.index, max_workers=10, load_livetracker=livetracker.load_livetracker
# )

if not download:
    st.caption(f"Selected {len(races)} races")
    st.caption("Checkbox 'load livetracker data' in sidebar to view race data")
    st.stop()

with st.spinner("Downloading livetracker data"):
    live_data, intermediates = select.get_races_livedata(races, max_workers=threads)

with st.expander("Filter livetracker data"):
    live_data = select.filter_livetracker(live_data)

st.subheader("Show livetracker")

live_data, PGMT = select.set_livetracker_PGMT(live_data)
distance_from_pace = "distance from PGMT"
facets = ["PGMT", "distance from PGMT", "split", "strokeRate"]

with st.spinner("Generating livetracker plot"):
    fig = plots.make_livetracker_plot(
        facets, *plots.melt_livetracker_data(live_data, 100), 
    )

with st.spinner("plotting..."):
    st.plotly_chart(fig, use_container_width=True)

if st.button("reset"):
    st.experimental_set_query_params()
    st.experimental_rerun()
else:
    state.update_query_params()
