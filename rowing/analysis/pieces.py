"""Pure piece-analysis computations for the rowing apps.

Crossing times, location timings, fastest times, stroke profiles and the
power/GPS interpolation. These contain no Streamlit calls and no caching, so
they can be imported and unit-tested directly; ``rowing.analysis.app`` exposes
thin ``@st.cache_data`` wrappers around them for the Streamlit layer.
"""

import logging

import pandas as pd

from rowing import utils
from rowing.analysis import geodesy, splits, telemetry

logger = logging.getLogger(__name__)


def get_crossing_times(gpx_data, locations=None, thresh=0.5):
    crossing_times, errors = utils.map_concurrent(
        splits.find_all_crossing_times,
        gpx_data,
        singleton=True,
        locations=locations,
        thresh=thresh,
    )
    if errors:
        logging.error(errors)

    return {k: d for k, d in crossing_times.items() if not d.empty}


def get_location_timings(gpx_data, locations=None, thresh=0.5):
    location_timings, errors = utils.map_concurrent(
        splits.get_location_timings,
        gpx_data,
        singleton=True,
        locations=locations,
        thresh=thresh,
    )
    if errors:
        logging.error(errors)
    return location_timings


def get_fastest_times(gpx_data):
    best_times, errors = utils.map_concurrent(
        splits.find_all_best_times,
        gpx_data,
        singleton=True,
    )
    if errors:
        logging.error(errors)
    return best_times


def make_stroke_profiles(telemetry_data, piece_data, nres=101):
    profiles = {}
    boat_profiles = {}
    crew_profiles = {}
    for piece, piece_times in piece_data["Timestamp"].iterrows():
        name, leg = piece[1:3]
        profile = telemetry_data[name]["Periodic"].sort_index(axis=1)
        start_time = piece_times.min()
        finish_time = piece_times.max()

        piece_profile = (
            profile[profile.Time.dt.tz_localize(None).between(start_time, finish_time)]
            .set_index("Time")
            .dropna(axis=1, how="all")
        )

        profiles[name, leg] = profile = telemetry.norm_stroke_profile(piece_profile, nres)
        gate_angle = profile.GateAngle
        gate_angle0 = gate_angle - gate_angle.values.mean(0, keepdims=True)
        for (pos, side), angle0 in gate_angle0.items():
            profile["GateAngle0", pos, side] = angle0

        mean_profile = profile.groupby(level=1).mean().reset_index().rename({"": "Boat"}, axis=1, level=1)
        boat_profiles[name, leg] = boat_profile = mean_profile.xs("Boat", axis=1, level=1).droplevel(axis=1, level=1)

        profile = (
            mean_profile[mean_profile.columns.levels[0].difference(boat_profile.columns)]
            .rename_axis(columns=("Measurement", "Position", "Side"))
            .stack(level=[1, 2], future_stack=True)
            .reset_index(["Position", "Side"])
            .rename_axis(index="Normalized Time")
            .reset_index()
        )
        crew_profiles[name, leg] = profile

    crew_profile = pd.concat(crew_profiles, names=["name", "leg"]).reset_index(["name", "leg"])
    crew_profile["Rower"] = crew_profile.name + "|" + crew_profile.Position

    return {
        "profiles": profiles,
        "boat_profiles": boat_profiles,
        "crew_profiles": crew_profiles,
        "crew_profile": crew_profile,
    }


def interpolate_power(telemetry_data, dists=0.005, n_iter=10):
    power_gps_data = {}
    for k, data in telemetry_data.items():
        gps = data["positions"]
        power = data["power"]

        power_gps = gps.set_index("time")[["longitude", "latitude"]].apply(utils.interpolate_series, index=power.Time)
        power_gps = power.join(pd.concat({("boat", ""): power_gps}, axis=1).swaplevel(0, -1, axis=1)).sort_index(axis=1)
        power_gps_data[k] = geodesy.interp_dataframe(power_gps, dists, n_iter=n_iter)

    return power_gps_data
