
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

TIME_STR = "%Y-%m-%d %H%M%S"

logger = logging.getLogger(__name__)


def loc_time_key(latitude, longitude, time):
    lat = np.round(latitude, 4)
    lon = np.round(longitude, 4)
    time = pd.Timestamp(time)
    return (f"{lat:.4f},{lon:.4f}", time.strftime(TIME_STR))


def parse_history_path(path):
    path = Path(path)
    time = datetime.datetime.strptime(path.stem, TIME_STR)
    lat, lon = map(float, path.parent.name.split(","))
    return lat, lon, time


def count(gen):
    counter = itertools.count()
    collections.deque(zip(gen, counter), maxlen=0)
    return next(counter)


class WeatherClient(utils.CachedClient):
    history_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    reverse_url = "http://api.openweathermap.org/geo/1.0/reverse"

    def __init__(
            self, api_key=None, path="weather-data", map_kws=None,
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

    def read_path(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def all_weather_history(self):
        return (self.path / "history").glob("**/*.json")

    def weather_history(self):
        for f in self.all_weather_history():
            if os.stat(f).st_size > 2:
                yield f

    def missing_history(self):
        for f in self.all_weather_history():
            if os.stat(f).st_size == 2:
                yield f

    def n_missing_history(self):
        return count(self.missing_history())

    def n_history(self):
        return count(self.weather_history())

    def download_missing(self, n=10, **kwargs):
        history = [
            self.load_weather_history(*parse_history_path(path), **kwargs)
            for i, path in zip(range(n), self.missing_history())
        ]
        return pd.json_normalize(history)

    def get_weather_history(self, latitude, longitude, time):
        key = ("history",) + loc_time_key(latitude, longitude, time)
        path = self.local_cache.get_path(key, self.path)
        if path.exists():
            data = self.read_path(path)
            if data:
                data.update(data.pop("data")[0])
                data.update(data.pop("weather")[0])

            return data

        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump({}, f)
        return {}

    def load_weather_history(self, latitude, longitude, time, **kwargs):
        key = ("history",) + loc_time_key(latitude, longitude, time)
        data = self.cached(
            key, self.download_weather_history, latitude, longitude, time, **kwargs
        )
        data.update(data.pop("data")[0])
        data.update(data.pop("weather")[0])
        return data

    def download_weather_history(self, latitude, longitude, time, **params):
        timestamp = utils.to_timestamp(time, unit="1s")
        params = {
            "lat": latitude,
            "lon": longitude,
            "dt": timestamp,
            "appid": self.password
        }
        logger.debug("downloading data at %.4f, %.4f for %s",
                     latitude, longitude, time)
        r = requests.get(self.history_url, params=params)
        r.raise_for_status()
        return r.json()

    def get_weather_histories(self, points=None):
        points = (
            map(parse_history_path, self.weather_history())
        )
        weather_data, errors = self.map_concurrent(
            self.get_weather_history,
            list(points)
        )
        return pd.json_normalize([data for data in weather_data if data])


def load_dtg_data():
    dtg_cols = [
        'Timestamp (GMT)',
        'Temperature (Celcius * 10)',
        'Humidity (%)',
        'Dew Point (Celcius * 10)',
        'Pressure (mBar)',
        'Mean wind speed (knots * 10)',
        'Average wind bearing (degrees)',
        'Sunshine (hours * 100)',
        'Rainfall (mm * 1000)',
        'Max wind speed (knots * 10)'
    ]
    dtg_data = pd.read_csv(
        "https://www.cl.cam.ac.uk/research/dtg/weather/weather-raw.csv",
        names=dtg_cols
    )
    dtg_data['datetime'] = \
        pd.to_datetime(dtg_data['Timestamp (GMT)'])
    dtg_data['timestamp'] = dtg_data.datetime.astype(int) // 1e9
    dtg_data['temperature'] = dtg_data['Temperature (Celcius * 10)'] / 10
    dtg_data['wind_speed'] = dtg_data['Mean wind speed (knots * 10)'] * \
        1852 / 3600
    dtg_data['bearing'] = dtg_data['Average wind bearing (degrees)']

    dtg_filtered = dtg_data[
        ~ dtg_data['Temperature (Celcius * 10)'].isin(
            [-400, -300, 400]
        )
    ].set_index("datetime")
    return dtg_filtered


def get_parser():
    parser = argparse.ArgumentParser(
        description='Download weather data'
    )
    parser.add_argument(
        '--api-key',
        type=str, nargs='?',
        help='api key'
    )
    parser.add_argument(
        '-c', '--credentials',
        type=str,
        default="weather_credentials.json",
        nargs='?',
        help='path to json file containing credentials (api-key)'
    )
    parser.add_argument(
        '--path',
        type=str, default="weather-data", nargs='?',
        help='folder path to download weather to'
    )
    parser.add_argument(
        'n',
        type=int, default=1, nargs='?',
        help='number of downloads'
    )
    utils.add_logging_argument(parser)
    return parser


def run(args):
    options = get_parser().parse_args(args)
    print(args)
    utils.set_logging(options)

    if options.credentials:
        weather_api = WeatherClient.from_credentials(
            options.credentials, path=options.path
        )
    elif options.api_key:
        weather_api = WeatherClient(options.api_key, path=options.path)
    else:
        raise argparse.ArgumentError("need credentials or api-key")

    history = weather_api.download_missing(options.n, reload=True)

    with pd.option_context('display.max_rows', None):
        print(history)

    n_missing = weather_api.n_missing_history()

    print(f"{n_missing} points to be downloaded")

    return history


def main():
    try:
        run(sys.argv[1:])
    except Exception as err:
        logger.error(err)

    input("Press enter to finish")


if __name__ == "__main__":
    main()
