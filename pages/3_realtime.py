
import logging
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm 
import streamlit as st

import plotly as pl
from plotly import express as px, subplots

from rowing.world_rowing import api, live, utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

competition = api.get_most_recent_competition()
races = api.get_races(competition_id=competition.competition_id)
race = races.iloc[-1]
race_id = race.race_id

race

facets = ["distanceFromLeader", "metrePerSecond", "strokeRate"]

get_live_race_data = st.cache_resource(live.LiveRaceData)

state = get_live_race_data(race_id, realtime_sleep=0.1, dummy=True)

fig_plot = st.empty()

with utils.ThreadPoolExecutor(max_workers=5) as executor, tqdm() as pbar:
    updates = utils.WorkQueue(executor, queue_size=1).run(
        state.update, state.tracker.run()
    )
    for i, (state, data) in enumerate(updates):
        pbar.update(1)
        plot_data = state.plot_data(facets)

        fig = px.line(
            plot_data, 
            x='distanceTravelled', 
            y='value', 
            color='boat', 
            facet_row="live", 
            hover_data=facets
        )
        fig.update_yaxes(
            matches=None
        )
        with fig_plot:
            st.plotly_chart(fig)

state.tracker.realtime_history

state.tracker.livetracker_history