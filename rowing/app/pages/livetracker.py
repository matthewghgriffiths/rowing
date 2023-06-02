

import logging

import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px

from rowing.world_rowing import api, livetracker, utils
from rowing.app import select, inputs, state

logger = logging.getLogger(__name__)

state.reset()

st.title("Show livetracker data for races")

competition = select.select_competition()
competition_id = competition.name
competition_type = competition.WBTCompetitionType

print(competition)

RACE_COL = [
        'race.DisplayName', 
        'race.Phase',
        'race.Gender', 
        'race.Category', 
        'race.raceStatus.DisplayName', 
        'boatClass.DisplayName',
        'race.Date', 
]

races = select.select_races(competition_id)

race = select.select_race(races)

cbts = select.select_best_times([race['boatClass.DisplayName']])
gmts = select.set_gmts(cbts)
gmt = gmts[race['boatClass.DisplayName']].total_seconds()
st.write(f"GMT set to {utils.format_totalseconds(gmt)}")

st.write(race)

tracker = livetracker.RaceTracker(race['race.id'], gmt=gmt)

live_data, intermediates = tracker.update_livedata()
boat_data = live_data.stack().reset_index()

boat_data['split'] = 500 / boat_data.metrePerSecond
boat_data['split min/500'] =  pd.Timestamp(0) + pd.to_timedelta(boat_data['split'], 's')

split = boat_data['split min/500']
pad = pd.Timedelta(10, 's')
sel = (boat_data.distanceTravelled > 100) & (boat_data.distanceTravelled < 2000) 
min_split = split[sel].min() - pad
max_split = split[sel].max() + pad


gmt_fig = px.line(
    boat_data, 
    x=boat_data.distanceTravelled, 
    y=boat_data.PGMT, 
    hover_data=[
        'distanceFromLeader', 'timeFromLeader', 'metrePerSecond', 'strokeRate', 'time',
    ],
    color='boat'
)
st.plotly_chart(gmt_fig)

fig = px.line(
    boat_data, 
    x=boat_data.distanceTravelled, 
    y=boat_data.timeFromLeader, 
    hover_data=[
        'distanceFromLeader', 'timeFromLeader', 'metrePerSecond', 'strokeRate', 'time',
    ],
    color='boat'
)
st.plotly_chart(fig)


fig = px.line(
    boat_data, 
    x='distanceTravelled', 
    y='split min/500', 
    hover_data=[
        'distanceFromLeader', 'timeFromLeader', 'metrePerSecond', 'strokeRate', 'time',
    ],
    color='boat',
    range_y=[min_split, max_split]
)
fig.update_layout(
    yaxis = dict(
        tickformat = "%M:%S.%f",   
    ),
)
st.plotly_chart(fig)

# state.reset()
state.update_query_params()
