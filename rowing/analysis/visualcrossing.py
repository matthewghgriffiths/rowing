import argparse
import sys
import json
import os
import datetime
from pathlib import Path
import logging
import collections
import itertools

import numpy as np
import pandas as pd
import requests

from rowing.analysis import splits, utils

DATETIME_STR = "%Y-%m-%dT%H:%M:%S"
TIME_STR = "%H:%M:%S"
DAY_STR = "%Y-%m-%d"

logger = logging.getLogger(__name__)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def loc_key(latitude, longitude):
    lat = np.round(latitude, 4)
    lon = np.round(longitude, 4)
    return f"{lat:.4f},{lon:.4f}"


def loc_hour_key(latitude, longitude, start_time, finish_time):
    start = pd.Timestamp(start_time).strftime(TIME_STR)
    finish = pd.Timestamp(finish_time).strftime(TIME_STR)

    return (loc_key(latitude, longitude), f"{start}-{finish}")


def loc_minutes_key(latitude, longitude, start_time, finish_time, minutes):
    lat = np.round(latitude, 4)
    lon = np.round(longitude, 4)
    start = pd.Timestamp(start_time).strftime(TIME_STR)
    finish = pd.Timestamp(finish_time).strftime(TIME_STR)

    return (f"{lat:.4f},{lon:.4f}", f"{start}-{finish}:{minutes}")


class WeatherClient(utils.CachedClient):
    history_url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/"
        "rest/services/weatherdata/history"
    )

    def __init__(
            self, api_key=None, path="visualcrossing-data", map_kws=None,
    ):
        super().__init__(
            username=None,
            password=api_key,
            path=path,
            local_cache=utils.json_cache,
            map_kws=map_kws
        )

    @classmethod
    def from_credentials(cls, credentials, **kwargs):
        with open(credentials, "r") as f:
            data = json.load(f)
            data.update(kwargs)

        return cls(**data)

    def load_hourly_weather(
            self, latitude, longitude, start_time, end_time, **kwargs):
        loc_key = loc_hour_key(latitude, longitude, start_time, end_time)
        key = ("hourly",) + loc_key

    def _hourly_weather_params(
        self, latitude, longitude, start_time, end_time, **params
    ):
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)

        params.update({
            "locations": loc_key(latitude, longitude),
            "startDateTime": start.date().strftime(DATETIME_STR),
            "endDateTime": end.date().strftime(DATETIME_STR),
            'dayStartTime': start.time().strftime(TIME_STR),
            'dayEndTime': end.time().strftime(TIME_STR),
            "key": self.password,
            "unitGroup": params.get("unitGroup", "base"),
            "contentType": params.get("contentType", "csv"),
            "combinationMethod": params.get("combinationMethod", "aggregate"),
            "aggregateHours": params.get("aggregateHours", 1),
        })
        return params

    def get_hourly_weather(
        self, latitude, longitude, start_time, end_time, request_kws=None, **params
    ):
        params = self._hourly_weather_params(
            latitude, longitude, start_time, end_time, **params)
        return requests.get(
            self.history_url, params=params, **(request_kws or {})
        )

    def _minutes_weather_params(
        self, latitude, longitude, start_time, end_time, minutes, **params
    ):
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        startDT = start.strftime(TIME_STR)
        endDT = end.strftime(TIME_STR)

        params.update({
            "location": loc_key(latitude, longitude),
            "startDateTime": startDT,
            "endDateTime": endDT,
            "aggregateMinutes": minutes,
            "key": self.password,
            "unitGroup": params.get("unitGroup", "base")
        })
        return params
