
import logging
import datetime
import time
from functools import partial, wraps

import streamlit as st
import pandas as pd

from rowing.world_rowing import api, utils, live, fields
from rowing.app import state, inputs, threads

logger = logging.getLogger(__name__)

get_live_race_data = st.cache_resource(live.LiveRaceData)

USE_CACHE = True


def cache_data(func=None, **kwargs):
    if func:
        cached_func = st.cache_data(func, **kwargs)

        @wraps(func)
        def new_func(*args, **kwargs):
            if USE_CACHE:
                logger.debug("using cache for %s", func)
                return cached_func(*args, **kwargs)
            else:
                logger.debug("not using cache for %s", func)
                return func(*args, **kwargs)

        return new_func
    else:
        return partial(cache_data, **kwargs)


@st.cache_data(persist=True)
def get_competitions(**kwargs):
    return api.get_competitions(**kwargs)


@st.cache_data(persist=False, ttl=600)
def get_races(competition_id):
    logger.debug("get_races(%s)", competition_id)
    return api.get_races(competition_id=competition_id).sort_values(
        fields.race_Date, ascending=False
    )


@st.cache_data(persist=True)
def get_events(competition_id):
    logger.debug("get_events(%s)", competition_id)
    return api.get_events(competition_id=competition_id)


@st.cache_data(persist=False, ttl=600)
def get_results(competition_id):
    logger.debug("get_results(%s)", competition_id)
    return api.get_intermediate_results(competition_id=competition_id)


@st.cache_data(persist=True)
def get_boat_classes():
    return api.get_boat_classes()


@st.cache_data(persist=False, ttl=600)
def get_entries(competition_id):
    events = api.get_events(
        competition_id, include="boats.boatAthletes.person")
    if events.empty:
        return

    boats = pd.concat({
        e: pd.json_normalize(boats)
        for e, boats in events.set_index(["Event"]).event_boats.items()
    }, names=['event']).reset_index(level=0).reset_index(drop=True)
    comp_athletes = pd.concat({
        e: pd.json_normalize(athletes)
        for e, athletes in boats.set_index("id").boatAthletes.items()
    }, names=['boat_id']).reset_index(drop=True)

    comp_athletes['Position'] = comp_athletes.athletePosition
    comp_athletes.loc[
        comp_athletes.athletePosition == 'b',
        'Position'
    ] = '1'
    comp_athletes.loc[
        comp_athletes.athletePosition == 's',
        'Position'
    ] = boats.set_index("id").boatAthletes.loc[
        comp_athletes.boatId
    ].map(len).replace({
        5: 4, 9: 8
    }).values[
        comp_athletes.athletePosition == 's',
    ].astype(str)
    comp_athletes.loc[
        comp_athletes.athletePosition == 'c',
        'Position'
    ] = 'cox'

    comp_boat_athletes = comp_athletes.reset_index().rename(
        columns={
            # 'athletePosition': 'Position',
            'person.DisplayName': 'Athlete',
        }
    ).join(
        boats.set_index('id').rename(columns={
            'event': 'Event',
            'DisplayName': 'Boat',
        }),
        on='boatId', rsuffix='_'
    )
    return comp_boat_athletes


@st.cache_data(persist=True)
def get_competition_boat_classes(competition_id):
    logger.debug("get_competition_boat_classes(%s)", competition_id)
    events = get_events(competition_id)
    boat_classes = get_boat_classes()
    event_boat_classes = boat_classes[
        boat_classes[fields.boatClass_id].isin(
            events[fields.event_boatClassId])
    ][fields.boatClass].sort_values()
    return event_boat_classes


@st.cache_data(persist=False, ttl=24 * 3600)
def get_cbts(boat_classes=None):
    cbts = api.get_competition_best_times()
    if boat_classes is None:
        return cbts
    return cbts[cbts[fields.bestTimes_BoatClass].isin(boat_classes)]


@st.cache_data(persist=True)
def load_livetracker(race_id, cached=True):
    logger.debug("load_livetracker(%s)", race_id)
    return live.load_livetracker(race_id, cached=cached)


@st.cache_data(persist=False, ttl=24 * 3600)
def get_races_livedata(races, max_workers=10):
    logger.debug("get_races_livedata(race_ids[%d])", len(races))
    live_data, intermediates, lane_info = live.get_races_livetracks(
        races.index, max_workers=max_workers, load_livetracker=load_livetracker
    )
    if live_data.empty:
        return live_data, intermediates, lane_info

    live_data = live_data.join(
        races[[
            fields.race_Date, fields.Race, fields.race_event,
            fields.Gender, fields.Category, fields.Phase,
            fields.boatClass, fields.GMT
        ]],
        on=fields.live_raceId,
        lsuffix='',
        rsuffix="_1"
    )
    live_data[fields.race_Date] = pd.to_datetime(live_data[fields.race_Date])
    live_data[fields.crew] = (
        live_data[fields.raceBoats] + " " + live_data[fields.boatClass])
    return live_data, intermediates, lane_info


get_realtime_race_data = st.cache_resource(live.LiveRaceData)

COMPETITION_COL = [
    fields.Competition,
    fields.started,
    fields.competition_venue,
    fields.competition_StartDate,
    fields.competition_Year,
    fields.competition_competitionType,
    fields.finished,
    "competition_id",
]


def select_competition(current=True, start_date=None, end_date=None, fisa=False):
    logger.debug("select_competition(current=%s)", current)
    st.write(
        """
        The most recent FISA competition will be loaded by default, 
        'select other competition' will allow you to choose older competitions. 
        """)
    current = inputs.modal_button(
        "select other competition",
        "Use current competition",
        key="current_competition",
        mode=current,
    )

    if current:
        competition_id = st.text_input(
            "Competition id:", api.get_most_recent_competition(
                fisa=fisa).competition_id
        )
        competition = api.get_worldrowing_record(
            "competition", competition_id, include="competitionType,venue"
        )
        competition.loc["started"] = pd.to_datetime(
            competition[fields.competition_StartDate]) < datetime.datetime.now()
        competition.loc["finished"] = pd.to_datetime(
            competition[fields.competition_StartDate]) < datetime.datetime.now()
    else:
        today = datetime.date.today()
        start_date = pd.to_datetime(
            start_date or state.get("competition.start_date", today),
            errors='coerce')
        end_date = pd.to_datetime(
            end_date or state.get("competition.end_date"),
            errors='coerce'
        )
        start_date = today if pd.isna(start_date) else start_date
        end_date = (
            today - datetime.timedelta(days=365*2) if pd.isna(end_date) else end_date)
        date_input = st.date_input(
            "Select date range to load",
            value=[end_date, start_date]
        )
        if len(date_input) == 2:
            end_date = date_input[0].isoformat()
            start_date = date_input[1].isoformat()
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

        competitions['started'] = \
            pd.to_datetime(
                competitions[fields.competition_StartDate]) < datetime.datetime.now()
        competitions['finished'] = \
            pd.to_datetime(
                competitions[fields.competition_EndDate]) < datetime.datetime.now()
        competitions = inputs.filter_dataframe(
            competitions[COMPETITION_COL],
            default=["started"],
            started=[True],
            key='competition',
            filters=False
        ).sort_values(fields.competition_StartDate, ascending=False)

        if competitions.empty:
            st.caption("No competition selected")
            st.stop()

        competition_id = state.get("CompetitionId")

        competition = inputs.select_dataframe(competitions, "competition")

    competition.loc[fields.WBTCompetitionType] = api.COMPETITION_TYPES.get(
        competition[fields.competition_competitionType],
        competition[fields.competition_competitionType]
    )
    # competition.index.name = 'CompetitionId'
    st.write(competition.loc[COMPETITION_COL + [fields.WBTCompetitionType]])
    return competition


RACE_COL = [
    fields.boatClass,
    fields.race_event,
    fields.Race,
    fields.Phase,
    fields.Gender,
    fields.Category,
    fields.Day,
    fields.race_Date,
    fields.race_event_competition,
    fields.race_raceStatus,
]


def filter_races(
        races,
        filters=False, select_all=True, select_first=False,
        **kwargs
):
    logger.debug("filter_races(races[%d], filters=%s)", len(races), filters)

    boat_classes = get_boat_classes()
    races = pd.merge(
        races, boat_classes,
        left_on=fields.race_event_boatClassId,
        right_on=fields.boatClass_id,
        how='left',
        suffixes=("", "_1")
    )
    st.subheader("Filter races to look at")

    a_finals = races[
        (races[fields.Phase] == 'Final A')
        & (races[fields.race_raceStatus] == 'Official')
    ]
    phases = races[fields.race_raceStatus].unique() if a_finals.size else [
        'Final A']

    kwargs.setdefault(fields.Phase, phases)
    kwargs.setdefault(
        fields.Gender, ['Men', 'Women', 'Mixed'])
    kwargs.setdefault(
        fields.Category, ['Open', 'Lightweight', 'PR1', 'PR2', 'PR3'])
    kwargs.setdefault(
        fields.race_raceStatus, ["Official", "Unofficial"])
    # kwargs.setdefault(
    #     "default", [fields.Gender, fields.Category, fields.race_raceStatus])
    # kwargs.setdefault(
    #     fields.race_boatClass, "*"
    # )
    races = inputs.filter_dataframe(
        races,
        options=RACE_COL,
        categories={fields.Phase, fields.race_event, fields.boatClass},
        filters=filters,
        select_all=select_all,
        select_first=select_first,
        key='filter_races',
        **kwargs,
    ).reset_index(drop=True)
    races[fields.race_Date] = pd.to_datetime(races[fields.race_Date])
    return races


def select_races(
        competition_id=None,
        competition_container=None,
        races_container=None,
        **kwargs,
):
    logger.debug(
        "select_races(%r, filters=%s)", competition_id, kwargs.get("filters"))
    # with st.expander("Select competition", state.get("expander.filter_competition", False)):
    with competition_container or st.container():
        if competition_id is None:
            competition = select_competition()
            competition_id = competition.competition_id

    races = get_races(competition_id)

    # with st.expander("Filter races", state.get("expander.filter_races", False)):
    with races_container or st.container():
        races = filter_races(races, **kwargs)

    return races


def select_race(races):
    sel_race = st.selectbox(
        "select race to load", races[fields.Race],
    )
    race = races.loc[races[fields.Race] == sel_race]
    return race.iloc[0]


def wait_for_next_race(n=5):
    next_races = api.get_next_races(n)
    if not next_races.empty:
        st.write("next races:")
        st.dataframe(fields.to_streamlit_dataframe(next_races))

        cols = st.columns(2)
        if cols[0].checkbox("refresh until next race"):
            with cols[1]:
                countdown = st.empty()
                for t in range(10, -1, -1):
                    countdown.metric("Refresh in", f"{t} s")
                    time.sleep(1)
                st.rerun()
        if st.button("refresh"):
            st.rerun()

    st.write(
        "no live race could be loaded, "
        "check replay in sidebar to see race replay")


def select_live_race(replay=False, **kwargs):
    if replay:
        races = select_races(
            filters=True,
            select_all=True,
            default=[fields.race_raceStatus],
            **kwargs
        )
        with kwargs.get("select_race") or st.container():
            if races.empty:
                st.write("no live races could be loaded")
                st.stop()

            race = inputs.select_dataframe(races, fields.Race)
    else:
        races = api.get_live_races(fisa=False)
        if races.empty:
            wait_for_next_race(n=5)
            st.stop()

        race = inputs.select_dataframe(races, fields.Race)
        if race is None:
            st.write("no live race could be loaded")
            if st.checkbox("refresh until next"):
                time.sleep(10)
                st.rerun()

            if st.button("refresh"):
                st.rerun()
            st.stop()

    return race


RESULT_COLS = [
    fields.PGMT,
    fields.raceBoatIntermediates_ResultTime,
    fields.GMT,
    fields.boatClass,
    fields.raceBoats,
    fields.Day,
    fields.Event,
    fields.race_event_competition,
    fields.Race,
    fields.Phase,
    fields.raceBoatIntermediates_Rank,
    fields.raceBoats_Lane,
    fields.Distance,
    fields.race_Date,
]


def select_results(race_results, key='race_results', **kwargs):
    filtered = inputs.filter_dataframe(
        race_results[RESULT_COLS].sort_values("PGMT", ascending=False),
        key=key, **kwargs
    )
    return filtered


CBT_COLS = [
    fields.bestTimes_ResultTime,
    fields.bestTimes_BoatClass,
    fields.WBTCompetitionType,
    fields.bestTimes_Competition,
    fields.bestTimes_Venue,
    fields.bestTimes_Event,
    fields.bestTimes_Race,
    fields.bestTimes_Country,
    fields.bestTimes_Date,
]


def select_best_times(boat_classes=None, *competition_types):
    cbts = get_cbts(boat_classes)
    if cbts.empty:
        return cbts
    if boat_classes is None:
        boat_classes = cbts[fields.bestTimes_BoatClass].unique()

    pick = inputs.modal_button(
        "Select competition best times", "Use world best times", "pickCBT", mode=True
    )
    if not pick:
        filtered_cbts = inputs.filter_dataframe(
            cbts[CBT_COLS].sort_values(
                fields.bestTimes_ResultTime, ascending=True),
            default=[fields.WBTCompetitionType, fields.bestTimes_BoatClass],
            categories={fields.bestTimes_BoatClass},
            key='GMT',
            **{
                fields.WBTCompetitionType: ['Elite Overall', *competition_types],
                fields.bestTimes_BoatClass: boat_classes,
            },
        )
        cbts = cbts.loc[filtered_cbts.index]

    return cbts


def set_gmts(cbts, *competition_types):
    wbts = cbts.groupby(fields.bestTimes_BoatClass)[
        fields.bestTimes_ResultTime].min()
    col1, col2 = st.columns(2)
    gmts = {
        "GMT": wbts,
        "best time": wbts,
    }
    for competition_type in competition_types:
        gmts[f"{competition_type} best time"] = cbts[
            cbts[fields.bestTimes_CompetitionType] == competition_type
        ].groupby(fields.bestTimes_BoatClass)[fields.bestTimes_ResultTime].min()

    with col2:
        uploaded = inputs.upload_csv(
            "Upload GMTs csv", key='upload_csv', index_col=0
        )
        if uploaded is not None:
            uploaded = uploaded.rename(
                columns=dict(zip(uploaded.columns, ['GMT']))
            )
            uploaded_gmt = pd.to_timedelta(uploaded.iloc[:, 0], unit='s')
            gmts['Uploaded'] = uploaded_gmt
            gmts[fields.GMT].update(uploaded_gmt)

    with col1:
        gmt_set = pd.concat(gmts, axis=1).apply(
            lambda s: s.dt.total_seconds().apply(utils.format_totalseconds)
        )
        gmts = utils.read_times(st.data_editor(gmt_set)[fields.GMT])

    with col2:
        inputs.download_csv(gmts.dt.total_seconds().round(2), "GMTs")

    return pd.to_timedelta(gmts)


def set_competition_gmts(competition_id, competition_type=None):
    comp_boat_classes = get_competition_boat_classes(competition_id)
    st.write(
        """
        You can filter the best times by competition type or boat class

        You can upload your own best times for boat classes, or download them
        """
    )
    cbts = select_best_times(comp_boat_classes, competition_type)

    if cbts.empty:
        st.write("No GMTS selected")
        st.stop()

    gmts = set_gmts(cbts, competition_type)
    return gmts


def select_competition_results(
    competition_id, gmts, stop_if_empty=True, **kwargs
):
    st.write(
        """
        Filter which race results to show.

        By default it selects only results the 2000m travelled, 
        it is possible to show the results for different intermediates
        distances as well. 

        It is also possible to filter by day, position, event and other criteria. 
        """
    )
    races = get_races(competition_id)
    events = get_events(competition_id)
    results = api.extract_results(races)
    boat_classes = get_boat_classes()

    merged_results = api.merge_competition_results(
        results, races, events, boat_classes, gmts)
    results = select_results(
        merged_results, **kwargs
    )
    results[fields.crew] = \
        results[fields.raceBoats] + " " + results[fields.boatClass]
    results = results.set_index(fields.crew)
    results = results.join(
        results.groupby('Race').Distance.max().rename('Finish Distance'),
        on='Race'
    ).reset_index()

    results = results.join(
        results.loc[
            results['Finish Distance'] == results.Distance,
            [fields.crew, 'Race', 'Intermediate Time', 'Intermediate Position']
        ].rename(columns={
            'Intermediate Time': "Finish Time",
            'Intermediate Position': 'Finish Position',
        }).set_index([fields.crew, 'Race']),
        on=[fields.crew, 'Race'], how='inner'
    )
    if results.empty and stop_if_empty:
        st.write("no results loaded")
        st.stop()

    return results.drop_duplicates(
        subset=['Boat', 'Distance', 'Race']
    ).set_index(fields.crew)


LIVE_COLS = [
    fields.Phase,
    fields.Event,
    fields.Race,
    fields.lane_Rank,
    fields.lane_Lane,
    fields.Gender,
    fields.Category,
    fields.Gender,
    fields.boatClass,
    fields.live_raceBoatTracker_currentPosition,
    fields.Date,
]


def filter_livetracker(live_data):
    live_data = inputs.filter_dataframe(
        live_data, key='live_data',
        options=LIVE_COLS,
        # default=[fields.lane_Rank],
        select=False,
        categories={
            fields.Event,
            fields.boatClass,
        },
        **{
            fields.lane_Rank: pd.Series(
                live_data[fields.lane_Rank].unique()
            ).dropna().sort_values().to_list()
        }
    )
    # live_data[fields.split] = pd.to_datetime(
    #     500 / live_data[fields.live_raceBoatTracker_metrePerSecond],
    #     unit='s', errors='coerce')
    # live_data[fields.avg_split] = pd.to_datetime(
    #     500. / live_data.avg_speed, unit='s', errors='coerce')
    # live_data[fields.ResultTime] = pd.to_datetime(
    #     live_data[fields.ResultTime].dt.total_seconds(),
    #     unit='s', errors='coerce')
    return live_data


def set_livetracker_PGMT(live_data):
    col1, col2 = st.columns(2)
    with col1:
        st.caption("input %GMT for pace boat")
    with col2:
        PGMT = st.number_input(
            "input %GMT for pace boat",
            min_value=0.01, max_value=1.1, value=1.,
            label_visibility="collapsed",
            key='setPGMT'
        )

    gmt_speed = (
        live_data[fields.race_distance]
        / live_data[fields.GMT].dt.total_seconds()
    )
    gmt_distance = live_data[fields.live_time] * gmt_speed
    pace_distance = gmt_distance * PGMT

    distance = live_data[fields.live_raceBoatTracker_distanceTravelled]
    live_data[fields.distance_from_pace] = pace_distance - distance
    live_data[fields.PGMT] = distance / pace_distance

    return live_data, PGMT


def last_race_results(n=10, fisa=True, cached=False):
    races = api.get_last_races(n, fisa=fisa, cached=cached)
    race_boats = pd.json_normalize(
        sum(races.Boat, [])
    ).join(
        races.set_index('race_id').Race, on='raceId'
    )
    boat_name = race_boats.set_index(
        'id'
    )[
        ['DisplayName', 'Race', 'Lane']
    ].drop_duplicates().rename(
        columns={'DisplayName': "Boat"})
    intermediates = pd.json_normalize(
        sum(race_boats.raceBoatIntermediates, [])
    )
    if not intermediates.empty:
        intermediates = intermediates.join(boat_name, on='raceBoatId')
        intermediates['Distance'] = intermediates[
            'distance.DisplayName'].str.extract("([\d]+)")[0].astype(int)
        intermediates['Time'] = pd.to_timedelta(
            intermediates['ResultTime']).apply(utils.format_timedelta)
        intermediates['Intermediate'] = ''

    return races, race_boats, intermediates


def unstack_intermediates(intermediates, col='Time'):
    table = pd.concat([
        intermediates.groupby(
            ['Race', 'Lane', 'Intermediate']
        ).Boat.first().reset_index(),
        intermediates[
            ['Race', 'Lane', 'Distance', col]
        ].rename(columns={
            "Distance": "Intermediate",
            col: "Boat",
        })
    ]).groupby(
        ["Race", "Intermediate", "Lane"]
    ).Boat.first().unstack().sort_index(ascending=False).fillna("")
    return table.rename(index=str)
