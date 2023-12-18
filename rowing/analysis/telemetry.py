import io 

import numpy as np 
import pandas as pd 

from rowing.analysis import files, splits, geodesy 

def Unnamed(col):
    if str(col).startswith("Unnamed: "):
        return ''
    else:
        return col

def parse_powerline_text_data(raw_text_data, sep='\t', use_names=True):
    parts = raw_text_data.split("=====")
    split_data = {}
    for part in parts:
        split = part.split("\n", 1)
        if len(split) == 2:
            k, raw_text = split
            key = k.replace("\t", "").replace(sep, "").strip()
            if 'eriodic' in key:
                split_data[key] = pd.read_table(
                    io.StringIO(raw_text), 
                    header=[0, 1], 
                    low_memory=False,
                    sep=sep
                ).rename(columns=Unnamed)#.dropna(axis=1, how='all')
            elif 'Rig' in key:
                split_data[key] = pd.read_table(
                    io.StringIO(raw_text), 
                    header=None, skiprows=[0, 1], 
                    names=['Position', 'Side'],
                    low_memory=False, 
                    sep=sep, 
                ).rename(columns=Unnamed)#.dropna(axis=1, how='all')
            elif raw_text:
                split_data[key] = pd.read_table(
                    io.StringIO(raw_text), 
                    low_memory=False,
                    sep=sep, 
                ).rename(columns=Unnamed)#.dropna(axis=1, how='all')
    
    return parse_powerline_data(split_data, use_names=use_names)

def parse_powerline_excel(data, use_names=True):
    data_groups = data[data[0] == "====="]
    data_keys = data_groups[1] + data_groups[2].fillna("")
    split_data = {}
    for i0, i1 in zip(data_groups.index, np.r_[data_groups.index[1:], data.index[-1] + 1]):
        key = data_keys[i0]
        key_data = data.loc[i0+1:i1-1].dropna(axis=1, how='all').dropna(axis=0, how='all')
        if "eriodic" in key:
            key_columns = pd.MultiIndex.from_frame(
                key_data.iloc[:2].fillna("").T
            )
            key_data = key_data.iloc[2:].convert_dtypes()
            key_data.columns = key_columns
        elif not key_data.empty:
            key_columns = key_data.iloc[0]
            key_data = key_data.iloc[1:].convert_dtypes()
            key_data.columns = key_columns
            
        for c, vals in key_data.items():
            if pd.api.types.is_numeric_dtype(vals.dtype):
                key_data[c] = vals.astype(float)
        split_data[key] = key_data

    return parse_powerline_data(split_data, use_names=use_names)

def parse_powerline_data(split_data, use_names=True):
    crew_data = split_data['Crew Info']
    gps_data = split_data["Aperiodic0x8013"]
    power_data = split_data["Aperiodic0x800A"]
    gps_info = split_data['GPS Info'].iloc[0]

    split_data['Crew List'] = crew_list = crew_data.set_index("Position").Name
    crew_list[crew_list.isna()] = crew_list.index[crew_list.isna()]

    start_time = pd.to_datetime(
        gps_info.UTC, 
        exact=False, 
        format="%d %b %Y %H:%M:%S (%Z)"
    )
    start_date = pd.Timestamp(start_time.date())
    lat, lon = gps_info.Lat, gps_info.Lon
    lat_scale = (geodesy._AVG_EARTH_RADIUS_KM * 1000 * np.pi/180)
    long_scale = lat_scale * np.cos(np.deg2rad(lat))

    t_shift = (gps_data['UTC Time'].Boat - gps_data.Time).max()

    
    positions = pd.concat({
        "time":  pd.to_timedelta(gps_data['UTC Time'].Boat, unit='milli') + start_date,
        "longitude": gps_data.long / long_scale + lon, 
        "latitude": gps_data.lat / lat_scale + lat
    }, axis=1).dropna().droplevel(1, axis=1)

    positions = files.process_latlontime(positions)
    positions['timeDelta'] = (positions.time.shift(-1) - positions.time).fillna(pd.Timedelta(0))
    positions['metrePerSeconds'] = positions.distanceDelta * 1000 / positions['timeDelta'].dt.total_seconds()

    power = power_data.copy()
    power.Time = pd.to_datetime(
        pd.to_timedelta(power.Time + t_shift, unit='milli') + start_date
    )
    stroke_length = power.MaxAngle - power.MinAngle
    effect = stroke_length - power.CatchSlip - power.FinishSlip
    power = pd.concat([
        power, pd.concat({
            "Length": stroke_length, 
            "Effective": effect, 
        }, axis=1)
    ], axis=1)

    split_data['positions'] = positions
    if use_names:
        power = power.rename(columns=crew_list.to_dict(), level=1)
    split_data['power'] = power

    return split_data 