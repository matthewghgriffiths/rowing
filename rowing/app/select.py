
import logging
import datetime
from os import get_blocking

import streamlit as st
import pandas as pd

from rowing.world_rowing import api, utils, live
from rowing.app import state, inputs, threads

logger = logging.getLogger(__name__)

@st.cache_data(persist=True)
def get_competitions(**kwargs):
    return api.get_competitions(**kwargs)

@st.cache_data(persist=True)
def get_races(competition_id):
    logger.debug("get_races(%s)", competition_id)
    return api.get_competition_races(competition_id=competition_id).reset_index().add_prefix("race.")

@st.cache_data(persist=True)
def get_events(competition_id):
    logger.debug("get_events(%s)", competition_id)
    return api.get_competition_events(competition_id=competition_id).add_prefix("event.")

@st.cache_data(persist=True)
def get_results(competition_id):
    logger.debug("get_results(%s)", competition_id)
    return api.get_intermediate_results(competition_id=competition_id)

@st.cache_data(persist=True)
def get_boat_classes():
    return api.get_boat_types().reset_index().add_prefix("boatClass.")

@st.cache_data(persist=True)
def get_competition_boat_classes(competition_id):
    logger.debug("get_competition_boat_classes(%s)", competition_id)
    events = get_events(competition_id)
    boat_classes = get_boat_classes()
    event_boat_classes = boat_classes[
        boat_classes['boatClass.id'].isin(events['event.boatClassId'])
    ]["boatClass.DisplayName"].sort_values()
    return event_boat_classes

@st.cache_data(persist=True)
def get_cbts(boat_classes=None):
    cbts = api.get_competition_best_times()
    if boat_classes is None:
        return cbts
    return cbts[cbts.BoatClass.isin(boat_classes)]

@st.cache_data(persist=True)
def get_livedata(race_id, gmt=None):
    logger.debug("get_livedata(%s)", race_id)
    live_data, live_boat_data, intermediates, race_distance = \
        live.get_race_livetracker(race_id, gmt=gmt, live=False)
    return live_data, live_boat_data, intermediates, race_distance

@st.cache_data(persist=True)
def load_livetracker(race_id, cached=True):
    logger.debug("load_livetracker(%s)", race_id)
    return live.load_livetracker(race_id, cached=cached)

@st.cache_data(persist=True)
def get_races_livedata(races, max_workers=10):
    logger.debug("get_races_livedata(race_ids[%d])", len(races))
    live_data, intermediates = live.get_races_livetracks(
        races.index, max_workers=max_workers, load_livetracker=load_livetracker
    )
    live_data = live_data.join(
        races[[
            "race.Date", "race.DisplayName", "race.Gender", 
            "race.Category", "race.Phase", "boatClass.DisplayName", "GMT"
        ]], 
        on='raceId'
    )
    live_data['race.Date'] = pd.to_datetime(live_data['race.Date'])
    return live_data, intermediates


COMPETITION_COL = [
    "DisplayName", "started", "venue.DisplayName", 
    "StartDate", "Year", "competitionType.DisplayName", 
    "finished", 
]

def select_competition(current=True):
    logger.debug("select_competition(current=%s)", current)
    current = inputs.modal_button(
        "select other competition", 
        "Use current competition", 
        key="current_competition",
        mode=current, 
    )
    
    if current:
        competition_id = st.text_input(
            "Competition id:", api.get_most_recent_competition().name
        )
        competition = pd.json_normalize(api.get_worldrowing_data(
            "competition", competition_id, include="competitionType,venue"
        )).set_index("id").iloc[0]
        competition.loc["started"] = pd.to_datetime(competition.StartDate) < datetime.datetime.now()
        competition.loc["finished"] = pd.to_datetime(competition.StartDate) < datetime.datetime.now()
    else:
        today = datetime.date.today()
        start_date = pd.to_datetime(state.get("competition.start_date", today))
        end_date = pd.to_datetime(
            state.get("competition.end_date", today - datetime.timedelta(days=365*2))
        )
        date_input = st.date_input(
            "Select date range to load", 
            value=[end_date, start_date]
        )
        if len(date_input) == 2:
            end_date, start_date = date_input[0].isoformat(), date_input[1].isoformat()
            state.set("competition.end_date", end_date)
            state.set("competition.start_date", start_date)

        competitions = get_competitions(
            year=False, 
            fisa=False,
            filter=[
                ("StartDate", start_date),
                ("EndDate", end_date),
                ("HasResults", 1),
            ],
            filterOptions=[
                ("StartDate", "lessThanEqualTo"),
                ("EndDate", "greaterThanEqualTo"),
            ],
        )
        if competitions.empty:
            st.write("No competitions found")
            st.stop()

        
        competitions['started'] = pd.to_datetime(competitions.StartDate) < datetime.datetime.now()
        competitions['finished'] = pd.to_datetime(competitions.EndDate) < datetime.datetime.now()
        competitions = inputs.filter_dataframe(
            competitions[COMPETITION_COL], 
            default=["started"],
            started=[True], 
            key='competition',
            filters=False
        ).sort_values("StartDate")

        if competitions.empty:
            st.caption("No competition selected")
            st.stop()

        competition_id = state.get("CompetitionId")
        if competition_id in competitions.index:
            name = competitions.DisplayName[competition_id]
            index = int((competitions.DisplayName == name).values.nonzero()[0][0])
        else:
            index = int(competitions.started.values.nonzero()[0][-1])

        sel_competition = st.selectbox(
            "select competition to load", 
            competitions, 
            index=index
        )
        competition = competitions[competitions.DisplayName == sel_competition].iloc[0]

    competition.loc['WBTCompetitionType'] = api.COMPETITION_TYPES[
        competition['competitionType.DisplayName']
    ]
    competition.index.name = 'CompetitionId'
    st.write(competition.loc[COMPETITION_COL + ["WBTCompetitionType"]])
    return competition

RACE_COL = [
        'boatClass.DisplayName',
        'race.DisplayName', 
        'race.Phase',
        'race.Gender', 
        'race.Category', 
        'race.Day', 
        'race.Date', 
        'race.event.competition.DisplayName',
        'race.raceStatus.DisplayName', 
]

def filter_races(races, filters=False, select_all=True):
    logger.debug("filter_races(races[%d], filters=%s)", len(races), filters)

    boat_classes = get_boat_classes()
    races = pd.merge(
        races, boat_classes, 
        left_on='race.event.boatClassId', right_on="boatClass.id"
    )
    st.subheader("Filter races to look at")
    races = inputs.filter_dataframe(
        races, 
        options=RACE_COL, 
        default=[
            'race.Phase', 'race.Gender', 'race.Category', 'race.raceStatus.DisplayName'
        ],
        **{
            'race.Phase': ['Final A'], 
            'race.Gender': ['Men', 'Women', 'Mixed'], 
            'race.Category': ['Open', 'Lightweight', 'PR1', 'PR2', 'PR3'], 
            'race.raceStatus.DisplayName': ["Official"]
        },
        filters=filters,
        select_all=select_all
    ).reset_index(drop=True)
    races['race.Date'] = pd.to_datetime(races['race.Date'])
    return races


def select_races(competition_id=None, filters=False, select_all=True):
    logger.debug("select_races(%r, filters=%s)", competition_id, filters)
    with st.expander("Select competition", state.get("expander.filter_competition", False)):
        if competition_id is None:
            competition = select_competition()
            competition_id = competition.name

    races = get_races(competition_id)
    with st.expander("Filter races", state.get("expander.filter_races", False)):
        races = filter_races(races, filters=filters, select_all=select_all)
    
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
    'PGMT', 'ResultTime', 'GMT', 'boatClass', 'Country', "race.Day", 
    "event", 'competition', 'race', 'racePhase',  
    'Rank', 'Lane', 'distance', "race.Date", 
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
    if cbts.empty:
        return cbts
    

    pick = inputs.modal_button(
        "Select competition best times", "Use world best times", "pickCBT", mode=True
    )
    if not pick:
        filtered_cbts = inputs.filter_dataframe(
            cbts[cbt_cols].sort_values("time", ascending=True), 
            default=["CompetitionType", "BoatClass"],
            CompetitionType=['Elite Overall', *competition_types], 
            categories={"BoatClass"},
            BoatClass=boat_classes, 
            key='GMT', 
        )
        cbts = cbts.loc[filtered_cbts.index]

    return cbts

def set_gmts(cbts, *competition_types):
    wbts = cbts.groupby("BoatClass").ResultTime.min()
    col1, col2 = st.columns(2)
    gmts = {
        "GMT": wbts, 
        "best time": wbts,
    }
    for competition_type in competition_types:
        gmts[f"{competition_type} best time"] = cbts[
            cbts.CompetitionType == competition_type
        ].groupby("BoatClass").ResultTime.min()
    
    with col2:
        uploaded = inputs.upload_csv(
            "Upload GMTs csv", key='upload_csv', index_col=0
        )
        if uploaded is not None:
            uploaded = uploaded.rename(
                columns=dict(zip(uploaded.columns, ['GMT']))
            )
            uploaded_gmt = pd.to_timedelta(uploaded.GMT, unit='s')
            gmts['Uploaded'] = uploaded_gmt
            gmts['GMT'].update(uploaded_gmt)

    with col1:
        gmt_set = pd.concat(gmts, axis=1).dropna().apply(
            lambda s: s.dt.total_seconds().apply(utils.format_totalseconds)
        )
        gmts = utils.read_times(st.data_editor(gmt_set).GMT)

    with col2:
        inputs.download_csv(gmts.dt.total_seconds().round(2), "GMTs")

    return gmts

LIVE_COLS = [
    'race.DisplayName', 
    'race.Phase',
    'race.Gender', 
    'race.Category', 
    "boatClass.DisplayName",
    "Rank",
    'race.Date', 
]
def filter_livetracker(live_data):
    live_data = inputs.filter_dataframe(
        live_data, key='live_data', 
        options=LIVE_COLS, 
        default=["Rank"],
        select=False, 
        **{
            "Rank": sorted(live_data.Rank.unique())
        }
    )
    live_data['split'] = pd.to_datetime(
        500 / live_data.metrePerSecond, unit='s', errors='coerce') 
    live_data['avg split'] = pd.to_datetime(
        500. / live_data.avg_speed, unit='s', errors='coerce')
    live_data["ResultTime"] = pd.to_datetime(
        live_data.ResultTime.dt.total_seconds(), unit='s', errors='coerce')
    return live_data


def set_livetracker_PGMT(live_data):
    col1, col2 = st.columns(2)
    with col1:
        st.caption("input %GMT for pace boat")
    with col2:
        PGMT = st.number_input(
            "input %GMT for pace boat", 
            min_value=0.01, max_value=1.1, value=1., 
            label_visibility="collapsed"
        )

    gmt_speed = live_data.raceDistance / live_data.GMT 
    gmt_distance = live_data.time * gmt_speed
    pace_distance = gmt_distance * PGMT

    distance_from_pace = f"distance from PGMT"
    live_data[distance_from_pace] = pace_distance - live_data.distanceTravelled
    live_data['PGMT'] = live_data.distanceTravelled / pace_distance

    return live_data, PGMT