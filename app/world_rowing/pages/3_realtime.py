
import streamlit as st

import logging
import datetime
import time 

import sys 
import os
from pathlib import Path 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm

import plotly as pl
from plotly import express as px, subplots

DIRPATH = Path(__file__).resolve().parent
LIBPATH = str(DIRPATH.parent.parent.parent)
realpaths = [os.path.realpath(p) for p in sys.path]
if LIBPATH not in realpaths:
    sys.path.append(LIBPATH)
    print("adding", LIBPATH)

from rowing.world_rowing import api, live, utils, fields
from rowing.app import select, inputs, state, plots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

st.set_page_config(
    page_title="World Rowing Realtime Livetracker",
    layout='wide'
    # page_icon="👋",
)


def main(params=None):
    state.update(params or {})
    st.title("World Rowing Realtime Livetracker")
    st.write(
        """
        Allows the following of the livetracker data from a race in realtime.

        If there are no live races currently on, then you can check _replay_ in the sidebar
        to see replays of historical races. 
        """
    )

    state.reset_button()

    with st.sidebar:
        with st.expander("Settings"):
            realtime_sleep = st.number_input(
                "poll", 0., 10., state.get("poll", 3.), step=0.5
            )
            replay = st.checkbox(
                "replay race data", state.get("replay", False))
            replay_step = st.number_input(
                "replay step", 1, 100, state.get("replay_step", 10))
            replay_start = st.number_input(
                "replay step", 0, 1000, state.get("replay_start", 0))
            
            fig_params = plots.select_figure_params()
            clear = st.button("clear cache")
            if clear:
                st.cache_data.clear()

    kwargs = {}
    race_expander = st.expander("Select race", True)
    if replay:
        with race_expander:
            kwargs['select_race'], kwargs['races_container'], kwargs["competition_container"] = st.tabs([
                "Select Race", "Filter Races", "Select Competition", 
            ])

    with race_expander:
        race = select.select_live_race(replay, **kwargs)
        st.subheader("Loading race: ")
        st.write(race)

    st.subheader("Livetracker")
    
    live_race = select.get_live_race_data(
        race.race_id,
        realtime_sleep=realtime_sleep,
        replay=replay,
        replay_step=replay_step,
        replay_start=replay_start, 
    )
    show_intermediates = st.empty()
    completed = st.progress(0., "Distance completed")
    fig_plot = st.empty()

    pbar = tqdm(live_race.gen_data(
        live_race.update, 
        plots.live_race_plot_data,
        plots.make_plots,
    ))
    for fig, *_ in pbar:
        pbar.set_postfix(distance=live_race.distance)
        completed.progress(
            live_race.distance / live_race.race_distance, 
            f"Distance completed: {live_race.distance}m/{live_race.race_distance}m"
        )

        with show_intermediates:
            plots.show_lane_intermediates(live_race.lane_info, live_race.intermediates)
        
        if fig is not None:
            with fig_plot:
                fig = plots.update_figure(fig, **fig_params)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("no live data could be loaded")

    select.wait_for_next_race(n=5)
    state.update_query_params()
    return state.get_state()


if __name__ == "__main__":
    main()
