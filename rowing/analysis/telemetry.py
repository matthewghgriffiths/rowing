import io
import zipfile
import ast

import numpy as np
import pandas as pd

from rowing.analysis import files, geodesy, splits

FIELDS = [
    'Angle 0.7 F',
    'Angle Max F',
    'Average Power',
    'AvgBoatSpeed',
    'CatchSlip',
    'Dist/Stroke',
    'Drive Start T',
    'Drive Time',
    'FinishSlip',
    'Max Force PC',
    'MaxAngle',
    'MinAngle',
    'Length',
    'Effective',
    'Rating',
    'Recovery Time',
    'Rower Swivel Power',
    'StrokeNumber',
    'SwivelPower',
    'Work PC Q1',
    'Work PC Q2',
    'Work PC Q3',
    'Work PC Q4',
    'Work PC',
]
TIMING_FIELDS = [
    'drive_angle',
    'drive_finish_angle',
    'min_angle_time',
    'drive_start_time',
    'predrive_start_time',
    'catch_lag',
    'max_angle_time',
    'drive_finish_time',
    'finish_lag',
    'max_force',
    'max_force_time'
]


def Unnamed(col):
    if str(col).startswith("Unnamed: "):
        return ''
    else:
        return col


def parse_powerline_text_data(raw_text_data, sep='\t', use_names=True, with_timings=True):
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
                ).rename(columns=Unnamed)  # .dropna(axis=1, how='all')
            elif 'Rig' in key:
                split_data[key] = rig = pd.read_table(
                    io.StringIO(raw_text),
                    header=None, skiprows=[0, 1],
                    names=['Position', 'Side'],
                    usecols=[0, 1],
                    low_memory=False,
                    index_col=None,
                    sep=sep,
                )[['Position', 'Side']]  # .dropna(axis=1, how='all')
            elif raw_text:
                split_data[key] = pd.read_table(
                    io.StringIO(raw_text),
                    low_memory=False,
                    sep=sep,
                ).rename(columns=Unnamed)  # .dropna(axis=1, how='all')

    return parse_powerline_data(split_data, use_names=use_names, with_timings=with_timings)


def parse_powerline_excel(data, use_names=True, with_timings=True):
    data_groups = data[data[0] == "====="]
    data_keys = data_groups[1] + data_groups[2].fillna("")
    split_data = {}
    for i0, i1 in zip(data_groups.index, np.r_[data_groups.index[1:], data.index[-1] + 1]):
        key = data_keys[i0]
        key_data = data.loc[i0+1:i1 - 1].dropna(
            axis=1, how='all'
        ).dropna(axis=0, how='all')
        if "eriodic" in key:
            key_columns = pd.MultiIndex.from_frame(
                key_data.iloc[:2].fillna("").T.reset_index(drop=True)
            )
            key_data = key_data.iloc[2:].convert_dtypes()
            key_data.columns = key_columns
        elif 'Rig' in key:
            key_data = key_data.iloc[
                2:, :2
            ]
            key_data.columns = ['Position', 'Side']
        elif not key_data.empty:
            key_columns = key_data.iloc[0].reset_index(drop=True)
            key_data = key_data.iloc[1:].convert_dtypes()
            key_data.columns = key_columns

        for c, vals in key_data.items():
            if pd.api.types.is_numeric_dtype(vals.dtype):
                key_data[c] = vals.astype(float)
        split_data[key] = key_data.reset_index(drop=True)

    return parse_powerline_data(split_data, use_names=use_names, with_timings=with_timings)


def parse_powerline_data(split_data, use_names=True, with_timings=True):
    crew_data = split_data['Crew Info']
    gps_data = split_data["Aperiodic0x8013"]
    power_data = split_data["Aperiodic0x800A"]
    gps_info = split_data['GPS Info'].iloc[0]
    rig_info = split_data['Rig Info']

    crew_list = crew_data.set_index("Position").Name
    crew_list = crew_list.where(crew_list.notna(), crew_list.index).astype(str)
    split_data['Crew List'] = crew_list

    for fmt in [
        "%d %b %Y %H:%M:%S (%Z)",
        "%d %b %Y %H%M%S (%Z)"
    ]:
        try:
            start_time = pd.to_datetime(
                gps_info.UTC,
                exact=False,
                format=fmt
            )
            break
        except ValueError:
            pass
    else:
        start_time = pd.to_datetime(
            gps_info.UTC,
            # exact=False,
            format="mixed",
            dayfirst=True,
            errors='coerce'
        )

    # start_date = pd.Timestamp(start_time.date())
    # t_shift = (gps_data['UTC Time'].Boat - gps_data.Time).max()
    start_time = start_time.astimezone(None)
    lat, lon = gps_info.Lat, gps_info.Lon
    lat_scale = (geodesy._AVG_EARTH_RADIUS_KM * 1000 * np.pi/180)
    long_scale = lat_scale * np.cos(np.deg2rad(lat))

    split_data["Periodic"]['Time'] = start_time + pd.to_timedelta(
        split_data['Periodic'].Time, unit='milli')

    positions = pd.concat({
        "time": start_time + pd.to_timedelta(gps_data['Time'].squeeze(), unit='milli'),
        "longitude": gps_data.long.apply(
            pd.to_numeric, errors='coerce') / long_scale + lon,
        "latitude": gps_data.lat.apply(
            pd.to_numeric, errors='coerce') / lat_scale + lat
    }, axis=1).dropna().droplevel(1, axis=1)

    if not positions.empty:
        positions = files.process_latlontime(positions)
        positions['timeDelta'] = (
            positions.time.shift(-1) - positions.time).fillna(pd.Timedelta(0))
        positions['metrePerSeconds'] = positions.distanceDelta * \
            1000 / positions['timeDelta'].dt.total_seconds()

    power = power_data.copy()
    power.columns = set_rower_sides(power.columns, rig_info)
    power.Time = start_time + pd.to_timedelta(power.Time, unit='milli')
    stroke_length = power.MaxAngle - power.MinAngle
    effect = stroke_length - power.CatchSlip - power.FinishSlip
    split_data['power'] = power = pd.concat([
        power, pd.concat({
            "Length": stroke_length,
            "Effective": effect,
        }, axis=1)
    ], axis=1)

    split_data['positions'] = positions.sort_index(axis=1)
    split_data["Periodic"].columns = set_rower_sides(
        split_data["Periodic"].columns, rig_info)
    if use_names:
        split_data['power'] = power.rename(
            columns=crew_list.to_dict(), level=1)
        split_data["Periodic"] = split_data["Periodic"].rename(
            columns=crew_list.to_dict(), level=1).sort_index(axis=1)

    if with_timings:
        split_data['power'] = add_timings(split_data)

    return split_data


def add_timings(piece_data):
    power = piece_data['power']
    periodic = piece_data['Periodic']
    parameters = piece_data[
        'Parameter Info'
    ].set_index('Parameter').Value.infer_objects()
    catch_thresh = parameters.loc['Sweep Drive Start']
    finish_thresh = parameters.loc['Sweep Recovery']

    timings = calc_catch_timings(
        periodic, catch_thresh=catch_thresh, finish_thresh=finish_thresh)
    matches = match_stroke_timings(timings, power.Time.sort_values())
    return power.join(
        timings.loc[
            matches.index
        ].set_index(matches).drop(
            'Time', axis=1, level=0
        )
    ).sort_values("Time")


def calc_catch_timings(periodic: pd.DataFrame, catch_thresh=30, finish_thresh=15):
    raw_data = periodic.infer_objects()

    raw_data['stroke'] = (raw_data['Normalized Time'].diff() < -50).cumsum()
    raw_data['seconds'] = (
        raw_data.Time - raw_data.Time.min()).dt.total_seconds()
    stroke_start = pd.Series(np.interp(
        np.arange(raw_data.stroke.iloc[-1] + 1) * 100 + 50,
        raw_data['Normalized Time'].squeeze() + raw_data.stroke * 100,
        raw_data.seconds
    ), np.arange(raw_data.stroke.iloc[-1] + 1))
    raw_data['norm_seconds'] = raw_data['seconds'] - \
        stroke_start.loc[raw_data.stroke].values

    GateAngle = raw_data.GateAngle
    GateForceX = raw_data.GateForceX
    GateAngleVel = raw_data.GateAngleVel
    norm_seconds = raw_data.norm_seconds

    # Catch timings
    min_angle_time = (
        (GateAngleVel > 0).diff().fillna(0)
        & (GateAngle < 0)
    ).apply(
        norm_seconds.where, other=np.nan
    ).groupby(raw_data.stroke).min()

    drive_angle_sig = (
        (GateForceX > catch_thresh).diff().fillna(0)
        & (GateAngle < 0)
        & (GateAngleVel > 0)
    )
    drive_start_time = drive_angle_sig.apply(
        norm_seconds.where, other=np.nan
    ).groupby(raw_data.stroke).min()
    _drive_angle = GateAngle.where(
        drive_angle_sig, np.nan)
    drive_angle = _drive_angle.groupby(raw_data.stroke).min()
    predrive_start_time = (
        (GateAngle < _drive_angle.bfill(axis=0)).diff()
        & (GateAngleVel < 0)
    ).apply(
        norm_seconds.where, other=np.nan
    ).groupby(raw_data.stroke).min()

    # Finish Timings
    drive_angle_sig = (
        (GateAngleVel < 0).diff().fillna(0)
        & (GateAngle > 0)
    )
    max_angle_time = drive_angle_sig.apply(
        norm_seconds.where, other=np.nan
    ).groupby(raw_data.stroke).max()
    finish_angle = GateAngle.where(
        drive_angle_sig, np.nan
    ).groupby(raw_data.stroke).max()
    drive_finish_time = (
        (GateForceX > finish_thresh).diff().fillna(0)
        & (GateAngle > 0)
        & (GateAngleVel > 0)
    ).apply(
        norm_seconds.where, other=np.nan
    ).groupby(raw_data.stroke).max()

    # Drive timings
    max_force = (
        GateForceX
        * (GateAngleVel > 0)
        * (GateForceX > catch_thresh)
    ).groupby(raw_data.stroke).max()
    max_force_time = (
        GateForceX == (
            GateForceX
            * (GateAngleVel > 0)
            * (GateForceX > catch_thresh)
        ).groupby(raw_data.stroke).transform('max')
    ).apply(
        norm_seconds.where, other=np.nan
    ).groupby(raw_data.stroke).min()

    timings = pd.concat({
        'drive_angle': drive_angle,
        'drive_finish_angle': finish_angle,
        'min_angle_time': min_angle_time,
        'drive_start_time': drive_start_time,
        'predrive_start_time': predrive_start_time,
        'catch_lag': drive_start_time - predrive_start_time,
        'max_angle_time': max_angle_time,
        'drive_finish_time': drive_finish_time,
        'finish_lag': max_angle_time - drive_finish_time,
        'max_force': max_force,
        'max_force_time': max_force_time,
    }, axis=1)
    timings['Time'] = raw_data['Time'].groupby(raw_data.stroke).min().values
    return timings


def match_stroke_timings(timings, stroke_start, thresh=0.1):
    time = timings.Time
    is_close = time.astype(bool) & 0
    match = time.astype(int) * 0

    for s in pd.to_timedelta([0, thresh, -thresh], unit='s'):
        match.loc[~is_close] = stroke_start.searchsorted(time[~is_close] + s)
        diff = (time[~is_close] -
                stroke_start.loc[match.loc[~is_close]].values)
        is_close.loc[~is_close] = diff.dt.total_seconds() < thresh * 2

    return match.loc[is_close].drop_duplicates(keep='last')


rename_scull_cols = {
    'Min Angle': 'MinAngle',
    'Swivel Power': 'SwivelPower',
}


def set_rower_sides(columns, rig_info):
    rower_sides = rig_info.set_index('Position').Side
    rower_sides.index = rower_sides.index.astype(str)
    power_columns = columns.to_frame()

    channels = power_columns.iloc[:, 0].str.extract(
        "^(?P<side>P |S )?(?P<channel>.*)")
    channels['rower'] = power_columns.iloc[:, 1]
    channels.side = channels.side.where(
        pd.notna, channels['rower'].replace(rower_sides)
    ).fillna("").replace({
        "P ": "Port", "S ": "Stbd"
    })
    channels['channel'] = channels['channel'].replace(
        rename_scull_cols
    )

    return pd.MultiIndex.from_frame(
        channels[['channel', 'rower', 'side']].fillna("")
    )


def interp_series(s, x, **kwargs):
    return pd.Series(
        np.interp(x, s.index, s, **kwargs), index=x
    )


def norm_stroke_profile(stroke_profile, n_res):
    stroke_norm_time = stroke_profile[
        ('Normalized Time', 'Boat', 'Boat')].squeeze()
    stroke = (
        stroke_norm_time.diff() < 0
    ).cumsum()
    return stroke_profile.set_index(
        stroke_norm_time.values
    ).groupby(
        stroke.values
    ).apply(
        lambda df:
        df.apply(
            interp_series, x=np.linspace(-50, 50, n_res)
        )
    ).rename_axis(
        index=("Stroke Count", "Normalized Time")
    )


def compare_piece_telemetry(telemetry_data, piece_data, gps_data, landmark_distances, window=0):
    telemetry_distance_data = {}
    for piece, distances in piece_data['Total Distance'].iterrows():
        name = piece[1]
        power = telemetry_data[name]['power']
        if window:
            time_power = power.set_index("Time").sort_index()
            avg_power = time_power.rolling(
                pd.Timedelta(seconds=window)
            ).mean()
            power = avg_power.reset_index()

        gps = gps_data[name]
        gps_adjusted_distance = np.interp(
            gps.distance,
            distances,
            landmark_distances,
            left=np.nan, right=np.nan
        )
        power_epoch = power['Time'].astype(int)
        power_adjusted_distance = np.interp(
            power_epoch,
            gps.time.astype(int),
            gps_adjusted_distance,
            left=np.nan, right=np.nan
        )

        power_distance = power.copy()  # .rename(columns={"Time": "Distance"})
        power_distance['Distance'] = power_adjusted_distance

        power_distance = power_distance[
            power_distance.Distance.notna()
        ].set_index("Distance")
        power_distance.columns.names = 'Measurement', 'Position', 'Side'
        telemetry_distance_data[piece] = power_distance

    compare_power = pd.concat(
        telemetry_distance_data,
        names=piece_data['Timestamp'].index.names
    ).stack(level=[1, 2], future_stack=True).reset_index()

    return compare_power


def get_interval_averages(piece_timestamps, telemetry_data):
    avg_telem, interval_telem = {}, {}
    for piece, timestamps in piece_timestamps.iterrows():
        _, name, leg = piece
        power = telemetry_data[name][
            'power'
        ].sort_values("Time").reset_index(drop=True)
        avgP, intervalP = splits.get_interval_averages(
            power.drop("Time", axis=1, level=0),
            power.Time,
            timestamps
        )
        for k in avgP.columns.remove_unused_levels().levels[0]:
            avg_telem.setdefault(k, {})[name, leg] = avgP[k].T
        for k in intervalP.columns.remove_unused_levels().levels[0]:
            interval_telem.setdefault(k, {})[name, leg] = intervalP[k].T

    piece_data = {}
    for k, data in avg_telem.items():
        piece_data[f"Average {k}"] = pd.concat(
            data, names=['name', 'leg', 'position', 'side'])
    for k, data in interval_telem.items():
        piece_data[f"Interval {k}"] = pd.concat(
            data, names=['name', 'leg', 'position', 'side'])

    return piece_data


def load_zipfile(file):
    telem_data = {}
    with zipfile.ZipFile(file) as z:
        for f in z.filelist:
            name, key = f.filename.removesuffix(
                ".parquet").split("/")
            data = pd.read_parquet(
                z.open(f.filename)
            )
            if data.columns.str.contains("\(").any():
                data.columns = data.columns.map(ast.literal_eval)
            telem_data.setdefault(name, {})[key] = data

    return telem_data
