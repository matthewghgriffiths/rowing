
from functools import partial

import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from . import utils 

dtype_checks = {
    is_timedelta64_dtype: "timedelta",
    is_datetime64_any_dtype: "datetime",
    is_numeric_dtype: "numeric",
    is_categorical_dtype: "categorical",
    is_object_dtype: "categorical",
}

def which_dtype(dtype):
    for check, dt in dtype_checks.items():
        if check(dtype):
            return dt 
    

field_names = {
    'PGMT': 'PGMT',
    "Time": "Time", 
    'GMT': 'GMT',
    'boatClass': 'Boat Class',
    'BoatClass': 'Boat Class',
    'boatClass_id': 'boatClass_id',
    'crew': 'Crew', 
    'raceBoats': 'Boat',
    'Day': 'Day',
    'Event': 'Event',
    'event_id': 'event_id', 
    'race_event_competition': 'competition',
    'Race': 'Race',
    'Phase': 'Phase',
    'Rank': 'Rank',
    'ResultTime': 'Finish Time',
    'split': 'Split', 
    'avg_split': 'Average Split', 
    'avg_speed': 'Average Speed', 
    'Date': 'Date',
    'raceBoats_raceBoatIntermediates': 'raceBoats_raceBoatIntermediates', 
    'raceBoatIntermediates_Rank': 'Intermediate Position',
    'raceBoatIntermediates_raceBoatId': 'raceBoatIntermediates_raceBoatId',
    'raceBoatIntermediates_distance': 'Intermediate Distance',
    'raceBoatIntermediates_ResultTime': 'Intermediate Time',
    'raceBoats_Lane': 'Lane',
    'raceBoats_id': 'raceBoats_id',
    'raceBoats_raceId': 'raceBoats_raceId', 
    'raceBoats_ResultTime': 'Finish Time',
    'Distance': 'Distance',
    'race_Date': 'Race Start',
    'race_id': 'race_id',
    'raceId': 'raceId', 
    'race_raceBoats': 'Race.Boat',
    'race_event': 'Race.Event',
    'race_eventId': 'race_eventId', 
    'race_event_boatClassId': 'race_event_boatClassId', 
    'race_competitionId': 'race_competitionId',
    'race_event_competitionId': 'race_event_competitionId',
    'race_event_RscCode': 'Rsc Code',
    'race_distance': 'Race Distance',
    # 'raceDistance': 'Race Distance'
    'race_raceStatus': 'Results Status',
    'boatClass_id': 'boatClass_id',
    'event_boatClassId': 'event_boatClassId',
    'Gender': 'Gender',
    'Category': 'Category',
    'race_boatClass': 'Boat Class',
    'competition_CompetitionCode': 'Competition Code',
    'Competition': 'competition',
    'competition_EndDate': 'Competition End Date',
    'competition_StartDate': 'Competition Start Date',
    'competition_competitionType': 'Competition Type',
    'CompetitionType': 'Competition Type',
    'competition_Year': 'Year',
    'competition_venue': 'Venue',
    'Country': 'Country',
    'Venue': 'Venue',
    'competition_venue_RegionCity': 'City',
    'competition_venue_Site': 'Site',
    'started': 'started',
    'finished': 'finished',
    'WBTCompetitionType': 'WBTCompetitionType',
    'live_distanceOfLeader': 'live_distanceOfLeader',
    'live_distanceOfLeaderFromFinish': 'live_distanceOfLeaderFromFinish',
    'live_raceBoatTracker_currentPosition': 'Current Position',
    'live_raceBoatTracker_distanceFromLeader': 'Distance from Leader',
    'live_raceBoatTracker_distanceTravelled': 'Distance',
    'live_distanceTravelled': 'Distance',
    'live_raceBoatTracker_id': 'live_raceBoatTracker_id',
    'live_raceBoatTracker_kilometrePersSecond': 'live_raceBoatTracker_kilometrePersSecond',
    'live_raceBoatTracker_metrePerSecond': 'Speed',
    'live_raceBoatTracker_raceBoatId': 'live_raceBoatTracker_raceBoatId',
    'live_raceBoatTracker_raceTrackerId': 'live_raceBoatTracker_raceTrackerId',
    'live_raceBoatTracker_startPosition': 'live_raceBoatTracker_startPosition',
    'live_raceBoatTracker_strokeRate': 'Stroke Rate',
    'live_raceId': 'live_raceId',
    'live_trackCount': 'live_trackCount',
    'live_time': 'Elapsed Time',
    'distance_from_pace': 'Distance from PGMT',
    'intermediates_Difference': 'Tntermediates Difference',
    'intermediates_Rank': 'Intermediate Position',
    'intermediates_ResultTime': 'Intermediate Time',
    'intermediates_StartPosition': 'intermediates_StartPosition',
    'intermediates_distanceId': 'intermediates_distanceId',
    'intermediates_distance_id': 'intermediates_distance_id',
    'intermediates_id': 'intermediates_id',
    'intermediates_raceBoatId': 'intermediates_raceBoatId',
    'intermediates_raceConfigId': 'intermediates_raceConfigId',
    'intermediates_raceConfig_code': 'intermediates_raceConfig_code',
    'intermediates_raceConfig_id': 'intermediates_raceConfig_id',
    'intermediates_raceConfig_position': 'intermediates_raceConfig_position',
    'intermediates_raceConfig_raceId': 'intermediates_raceConfig_raceId',
    'intermediates_raceConfig_value': 'intermediates_raceConfig_value',
    'lane_raceBoatAthletes': 'lane_raceBoatAthletes',
    'lane_WorldCupPoints': 'lane_WorldCupPoints',
    'lane__finished': 'lane__finished',
    'lane_countryId': 'lane_countryId',
    'lane_worldBestTimeId': 'lane_worldBestTimeId',
    'lane_Remark': 'lane_Remark',
    'lane_raceId': 'lane_raceId',
    'lane_ResultTime': 'Finish Time',
    'lane_id': 'lane_id',
    'lane_Rank': 'Position',
    'lane_Lane': 'Lane',
    'lane_invalidMarkResultCode': 'lane_invalidMarkResultCode',
    'lane_InvalidMarkResult': 'lane_InvalidMarkResult',
    'lane_DataTarget': 'lane_DataTarget',
    'lane_boatId': 'lane_boatId',
    'lane_country_id': 'lane_country_id',
    'lane_country': 'Country',
    'lane_country_CountryCode': 'Country Code',
    'lane_country_IsNOC': 'lane_country_IsNOC',
    'lane_country_IsFormerCountry': 'lane_country_IsFormerCountry',
}
field_names.update(
    (k.casefold(), v) for k, v in list(field_names.items())
)

locals().update(field_names)

field_types = {
    field_names[PGMT]: "percentage", 
}







def identity(x):
    return x 

def get_datatypes(data):
    dtypes = data.dtypes.map(which_dtype).to_dict()
    cols = data.columns[data.columns.isin(field_types)]
    dtypes.update(zip(cols, cols.map(field_types.get)))
    return dtypes 

def format_datatype(data, **formatters):
    datatypes = get_datatypes(data)
    return pd.concat({
        col: values.apply(formatters.get(datatypes[col], identity))
        for col, values in data.items()
    }, axis=1)

streamlit_formatters = {
    "timedelta": utils.format_timedelta, 
    "percentage": "{:,.2%}".format
}
to_streamlit_dataframe = partial(format_datatype, **streamlit_formatters)

def to_timestamp(s):
    return s + pd.Timestamp(0)

plotly_formatters = {
    "timedelta": to_timestamp
}
to_plotly_dataframe = partial(format_datatype, **plotly_formatters)


def rename_column(s, prefix=''):
    c = f"{prefix}_{s}".replace(".", "_")
    if c.endswith("DisplayName"):
        c = c[:-12]

    return field_names.get(c.casefold(), c)

def renamer(prefix): 
    return partial(rename_column, prefix=prefix)

TICKFORMATS = {
    "percentage": ",.0%", 
    "timedelta": "%-M:%S",
}
DATAFORMATS = {
    "percentage": ":.1%", 
    "timedelta": "|%-M:%S.%L",
}

def filter_numerical_columns(data):
    dtypes = data.dtypes
    return dtypes.index[
        dtypes.map(is_numeric_dtype)
        | dtypes.map(is_datetime64_any_dtype)
        | dtypes.map(is_timedelta64_dtype)
    ]

def filter_categorical_columns(data, max_unique=10):
    dtypes = data.dtypes
    nunique = data.apply(lambda s: s.nunique()) <= max_unique
    return dtypes.index[
        dtypes.map(is_object_dtype)
        | dtypes.map(is_categorical_dtype)
        | nunique
    ]

