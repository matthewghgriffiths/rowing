
from functools import partial

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
    'ResultTime': 'Time',
    'split': 'Split', 
    'avg_split': 'Average Split', 
    'Date': 'Date',
    'raceBoats_raceBoatIntermediates': 'raceBoats_raceBoatIntermediates', 
    'raceBoatIntermediates_Rank': 'Intermediate Position',
    'raceBoatIntermediates_raceBoatId': 'raceBoatIntermediates_raceBoatId',
    'raceBoatIntermediates_distance': 'Intermediate Distance',
    'raceBoatIntermediates_ResultTime': 'Intermediate Time',
    'raceBoats_Lane': 'Lane',
    'raceBoats_id': 'raceBoats_id',
    'raceBoats_raceId': 'raceBoats_raceId', 
    'raceBoats_ResultTime': 'Time',
    'Distance': 'Distance',
    'race_Date': 'Race Start',
    'race_id': 'race_id',
    'race_raceBoats': 'Race.Boat',
    'race_event': 'Race.Event',
    'race_eventId': 'race_eventId', 
    'race_event_RscCode': 'Rsc Code',
    'race_distance': 'Race Distance',
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
    'WBTCompetitionType': 'WBTCompetitionType'
}
field_names.update(
    (k.casefold(), v) for k, v in list(field_names.items())
)

locals().update(field_names)

field_types = {
    field_names["PGMT"]: "percentage", 
}

dtype_formatters = {
    "timedelta": utils.format_timedelta, 
    "percentage": "{:,.2%}".format
}

field_formatters = {
    k: dtype_formatters[v] for k, v in field_types.items()
}


def field_formats(data):
    dtypes = data.dtypes.map(which_dtype)
    formats = dtypes[
        dtypes.isin(dtype_formatters)
    ].replace(dtype_formatters).to_dict()
    cols = data.columns[data.columns.isin(field_types)]
    formats.update(zip(cols, cols.map(field_formatters.get)))
    return formats

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

