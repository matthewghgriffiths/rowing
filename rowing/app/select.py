
import datetime
from os import get_blocking

import streamlit as st
import pandas as pd

from rowing.world_rowing import api, utils
from rowing.app import state, inputs

@st.cache_data
def get_races(competition_id):
    return api.get_competition_races(competition_id=competition_id).reset_index().add_prefix("race.")

@st.cache_data
def get_events(competition_id):
    return api.get_competition_events(competition_id=competition_id).add_prefix("event.")

@st.cache_data
def get_results(competition_id):
    return api.get_intermediate_results(competition_id=competition_id)

@st.cache_data
def get_boat_classes():
    return api.get_boat_types().reset_index().add_prefix("boatClass.")



@st.cache_data
def get_competition_boat_classes(competition_id):
    events = get_events(competition_id)
    boat_classes = get_boat_classes()
    event_boat_classes = boat_classes[
        boat_classes['boatClass.id'].isin(events['event.boatClassId'])
    ]["boatClass.DisplayName"].sort_values()
    return event_boat_classes

# @st.cache_data
# def get_cbts(competition_id):
#     events = get_events(competition_id)
#     boat_classes = get_boat_classes()
#     event_boat_classes = boat_classes[
#         boat_classes['boatClass.id'].isin(events['event.boatClassId'])
#     ]["boatClass.DisplayName"].sort_values()
#     return get_boat_class_cbts(event_boat_classes)

@st.cache_data
def get_cbts(boat_classes=None):
    cbts = api.get_competition_best_times()
    if boat_classes is None:
        return cbts
    return cbts[cbts.BoatClass.isin(boat_classes)]


def select_competition():
    current = inputs.modal_button(
        "select other competition", 
        "select current competition", 
        key="current_competition",
    )
    competition_id = state.get("CompetitionId")
    if current:
        print(f"{competition_id=}")
        competition_id = st.text_input(
            "Competition id:", 
            competition_id or api.get_most_recent_competition().name
        )
        competition = pd.json_normalize(api.get_worldrowing_data(
            "competition", competition_id, include="competitionType,venue"
        )).set_index("id").iloc[0]
    else:
        today = datetime.date.today()
        year = st.number_input(
            "select year of competition", 
            1957, today.year, 
            value=state.get("year", today.year),
            step=1, 
        )
        state.set('year', year)

        competitions = api.get_competitions(year)
        competitions['started'] = pd.to_datetime(competitions.StartDate) < datetime.datetime.now()
        competitions['finished'] = pd.to_datetime(competitions.EndDate) < datetime.datetime.now()

        comp_cols = [
            "DisplayName", "started", "venue.DisplayName", 
            "StartDate", "Year", "competitionType.DisplayName", 
            "finished", 
        ]
        filtered_competitions = inputs.filter_dataframe(
            competitions[comp_cols], 
            default=["started"],
            started=[True], 
            key='competition',
            filters=False
        ).sort_values("StartDate")

        st.dataframe(filtered_competitions.reset_index(drop=True), use_container_width=True)

        if competition_id in competitions.index:
            name = competitions.DisplayName[competition_id]
            index = int((filtered_competitions.DisplayName == name).values.nonzero()[0][0])
        else:
            index = int(filtered_competitions.started.values.nonzero()[0][-1])

        sel_competition = st.selectbox(
            "select competition to load", 
            filtered_competitions, 
            index=index
        )
        competition = filtered_competitions[
            filtered_competitions.DisplayName == sel_competition
        ].iloc[0]

    competition.loc['WBTCompetitionType'] = api.COMPETITION_TYPES[
        competition['competitionType.DisplayName']
    ]
    return competition

RACE_COL = [
        'race.DisplayName', 
        'race.Phase',
        'race.Gender', 
        'race.Category', 
        'race.raceStatus.DisplayName', 
        'boatClass.DisplayName',
        'race.Date', 
]

def select_races(competition_id=None):
    if competition_id is None:
        competition = select_competition()
        competition_id = competition.name

    races = get_races(competition_id)
    boat_classes = get_boat_classes()
    races = pd.merge(
        races, boat_classes, 
        left_on='race.event.boatClassId', right_on="boatClass.id"
    )
    races = inputs.filter_dataframe(
        races, 
        options=RACE_COL, 
        default=['race.Phase', 'race.Category', 'race.raceStatus.DisplayName'],
        **{
            'race.Phase': ['Final A'], 
            'race.Gender': ['Men', 'Women', 'Mixed'], 
            'race.Category': ['Open', 'Lightweight', 'PR1', 'PR2', 'PR3'], 
            'race.raceStatus.DisplayName': ["Official"]
        }
    )
    return races 
    

def select_race(races):
    st.dataframe(
        races[RACE_COL]
    )
    sel_race = st.selectbox(
        "select race to load", 
        races['race.DisplayName'], 
        # index=index
    )
    race = races.loc[
        races['race.DisplayName'] == sel_race
    ]
    return race.iloc[0]


RESULT_COLS = [
    'PGMT', 'ResultTime', 'GMT', 'boatClass', 'Country', "race.Date", 
    "event", 'competition', 'race', 'racePhase',  
    'Rank', 'Lane', 'distance', 
]

def select_results(race_results, **kwargs):
    filtered = inputs.filter_dataframe(
        race_results[RESULT_COLS].sort_values("PGMT", ascending=False), 
        **kwargs
    )
    return filtered 


def select_best_times(boat_classes=None, *competition_types):
    cbts = get_cbts(boat_classes)

    cbt_cols = [
        'time', 'BoatClass', 'CompetitionType', 'Competition', 'Venue',
        'Event', 'Race', 'Country', 'Date', 
    ]

    pick = inputs.modal_button("Filter GMTs", "Use World Best", "pickGMT")
    if not pick:
        filtered_cbts = inputs.filter_dataframe(
            cbts[cbt_cols].sort_values("time", ascending=True), 
            default=["CompetitionType", "BoatClass"],
            CompetitionType=['Elite Overall', *competition_types], 
            categories={"BoatClass"},
            key='GMT', 
        )
        st.dataframe(filtered_cbts, use_container_width=True)
        cbts = cbts.loc[filtered_cbts.index]

    return cbts

def set_gmts(cbts, *competition_types):
    wbts = cbts.groupby("BoatClass").ResultTime.min()
    gmts = {
        "GMT": wbts, 
        "best time": wbts,
    }
    for competition_type in competition_types:
        gmts[f"{competition_type} best time"] = cbts[
            cbts.CompetitionType == competition_type
        ].groupby("BoatClass").ResultTime.min()
        
    gmt_set = pd.concat(gmts, axis=1).apply(
        lambda s: s.dt.total_seconds().apply(utils.format_totalseconds)
    )
    gmts = utils.read_times(st.experimental_data_editor(gmt_set).GMT)

    return gmts