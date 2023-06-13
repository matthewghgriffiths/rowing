
import os
import sys
import getpass
import logging
from datetime import datetime, timedelta
from typing import Optional
from io import BytesIO
import argparse
import json
import zipfile
import shutil
from pathlib import Path
import re

import numpy as np
import pandas as pd
import gpxpy
# import cloudscraper
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError,
)

from rowing.analysis.utils import (
    map_concurrent, unflatten_json, strfsplit,
    add_logging_argument, set_logging, _YMD, add_credentials_arguments
)
from rowing.analysis.files import parse_gpx_data, read_fit_zipfile, parse_fit_data, activity_data_to_excel
from rowing.analysis import splits, files, utils


logger = logging.getLogger(__name__)

GARMIN_EPOCH = pd.Timestamp('1989-12-31 00:00:00')
UNIX_EPOCH = pd.Timestamp(0)
GARMIN_TIMESTAMP = (GARMIN_EPOCH - UNIX_EPOCH) // pd.Timedelta('1s')


_API: Optional[Garmin] = None

_ACTIVITY_TYPES = {
    'cycling': {
        "activityType": "cycling",
    },
    'running': {
        "activityType": "running",
    },
    'rowing': {
        "activityType": "other",
        'activitySubType': 'rowing'
    }
}


class GarminClient(utils.CachedClient):
    def __init__(
            self, username=None, password=None,
            local_cache=None, path="ludum_data", map_kws=None,
    ):
        self.client = None
        super().__init__(
            username=username,
            password=password,
            path=path,
            local_cache=local_cache,
            map_kws=map_kws
        )

    def login(self, username=None, password=None, max_retries=5):
        username = username or self.username
        password = password or self.password
        self.client = login(username, password, max_retries=max_retries)
        return self

    def get(self, url, **kwargs):
        return self.client.modern_rest_client.get(url, **kwargs)

    def get_json(self, url, **kwargs):
        r = self.get(url, **kwargs)
        r.raise_for_status()
        return r.json()

    def cached_json(self, key, url, path=None, local_cache=None, reload=False, **kwargs):
        local_cache = local_cache or self.local_cache
        path = path or self.path
        if local_cache:
            if reload:
                return local_cache.update(
                    key, path, self.get_json, url, **kwargs
                )

            return local_cache.get(
                key, path, self.get_json, url, **kwargs
            )

        return self.get_json(url, **kwargs)

    def get_activities(self, start=None, end=None, period="7d", reload=False):
        end = end and pd.Timestamp(end)
        start = start and pd.Timestamp(start)
        period = period and pd.Timedelta(period)
        if not end and not start:
            end = pd.Timestamp.today().date()
        if end and not start:
            start = end - period
        elif start and not end:
            end = start + period

        dates = pd.date_range(start, end)
        activities, errors = self.map_concurrent(
            self.get_day_activities,
            dict(zip(dates, dates)),
            singleton=True,
            reload=reload
        )
        errors and logger.warning("get_activities had errors %s", errors)
        activities = pd.concat(
            map(pd.json_normalize, activities.values()), ignore_index=True)
        activities['time'] = pd.to_datetime(activities.startTimeLocal)
        return activities

    def get_day_activities(self, date, limit=1000, reload=False):
        date = pd.to_datetime(date).strftime("%Y-%m-%d")
        params = {'start': 0, 'limit': limit,
                  'startDate': date, 'endDate': date}
        url = self.client.garmin_connect_activities
        path = self.path / "activity"
        return self.cached_json(date, url, params=params, path=path, reload=reload)

    def load_fits(self, activity_ids):
        positions, errors = self.map_concurrent(
            self.load_fit,
            dict(zip(activity_ids, activity_ids)),
            singleton=True,
        )
        return pd.concat(positions, names=[''])

    def load_fit(self, activity_id):
        return parse_garmin_fit_json(self.load_fit_json(activity_id))

    def load_fit_json(self, activity_id):
        path = self.path / "fit"
        local_cache = self.local_cache
        if local_cache:
            return local_cache.get(
                activity_id, path, self.download_fit_json, activity_id
            )
        return self.download_fit_json(activity_id)

    def download_fit_json(self, activity_id):
        zip_data = self.client.download_activity(
            activity_id, dl_fmt=self.client.ActivityDownloadFormat.ORIGINAL)
        return files.zipfit_to_json(BytesIO(zip_data))

    def get_weather(self, activity_id, **kwargs):
        # temp in F
        # wind speed in mph
        return self.cached(
            ("weather", activity_id),
            self.client.get_activity_weather, activity_id, **kwargs
        )

    def get_stats(self, day, **kwargs):
        date = pd.Timestamp(day).date().isoformat()
        return self.cached(
            ("stats", date), self.client.get_stats, day, **kwargs
        )

    def get_heart_rates(self, day, **kwargs):
        date = pd.Timestamp(day).date().isoformat()
        return self.cached(
            ("heart_rate", date), self.client.get_heart_rates, day, **kwargs
        )

    def get_rhr_day(self, day, **kwargs):
        date = pd.Timestamp(day).date().isoformat()
        return self.cached(
            ("rhr", date), self.client.get_rhr_day, day, **kwargs
        )

    def get_sleep_data(self, day, **kwargs):
        date = pd.Timestamp(day).date().isoformat()
        return self.cached(
            ("sleep", date), self.client.get_sleep_data, day, **kwargs
        )

    def get_stress_data(self, day, **kwargs):
        date = pd.Timestamp(day).date().isoformat()
        return self.cached(
            ("stress", date), self.client.get_stress_data, day, **kwargs
        )

    def get_heart_rate_values(self, start=None, end=None, periods=None):
        heart_rate_data, errors = utils.map_concurrent(
            self.get_heart_rates,
            list(pd.date_range(start, end, periods, freq='D')),
            singleton=True
        )
        hr_values = (hr['heartRateValues'] for hr in heart_rate_data)
        heart_rates = pd.DataFrame(sum(
            (hr for hr in hr_values if hr), [])
        )
        heart_rates.columns = ["timestamp", "heart_rate"]
        heart_rates["time"] = pd.to_datetime(heart_rates.timestamp, unit='ms')
        return heart_rates

    def get_hourly_heart_rates(self, start=None, end=None, periods=None):
        heart_rates = self.get_heart_rate_values(start, end, periods)

        hourly_heart_rates = heart_rates.groupby(
            pd.Grouper(key="time", freq='H')
        ).heart_rate.mean()
        hourly_heart_rates.index = pd.MultiIndex.from_arrays([
            hourly_heart_rates.index.date, hourly_heart_rates.index.hour
        ])
        hourly_heart_rates = hourly_heart_rates.unstack()
        hourly_heart_rates.columns = [
            f"{h:02d}:00" for h in hourly_heart_rates.columns]
        return hourly_heart_rates


def merge_index(index, activities):
    return pd.MultiIndex.from_frame(
        index.to_frame().reset_index(drop=True).join(
            activities[
                ['activityId', 'activityName', "time", 'activityType.typeKey']
            ].set_index("activityId").rename(columns={
                'activityName': "name",
                'activityType.typeKey': "activity",
            }),
            on="activity_id"
        )[["activity", "time", "name", "data"]]
    )


def find_best_times(positions, activities, max_distance=20, max_group=4, max_order=20):
    session_positions = positions.groupby(level=0)
    session_best_times, errors = utils.map_concurrent(
        splits.find_all_best_times,
        session_positions,
        singleton=True,
        total=session_positions.ngroups,
        cols=["bearing", "cadence", "heart_rate"],
        threaded=False,
    )
    best_times = pd.concat(
        session_best_times,
        names=['activity_id', 'length', 'distance']
    ).reset_index()
    best_times.columns.name = "data"

    best_times['order'] = best_times.groupby(
        ["activity_id", "length"]).cumcount() + 1
    best_times = best_times.set_index([
        "activity_id", "length", "order"
    ]).unstack(level=[1, 2]).T.unstack(level=0)

    best_times.columns = merge_index(best_times.columns, activities)

    distance_orders = (
        (s, d, i * max_group + j) for i in range(max_order // max_group)
        for s, d in splits._STANDARD_DISTANCES.items() for j in range(max_group)
    )
    distance_orders = pd.MultiIndex.from_tuples([
        (s, n + 1) for s, d, n in distance_orders
        if n * d <= max_distance
    ], names=["interval", "rank"])

    return best_times.sort_index(
        axis=1, ascending=[False] + [True] * (best_times.columns.nlevels - 1)
    ).reindex(distance_orders)


def parse_garmin_fit_json(fit_json):
    positions = pd.DataFrame.from_records(fit_json)
    positions['time'] = pd.to_datetime(
        positions.timestamp + GARMIN_TIMESTAMP, unit='s'
    )
    positions['distance'] /= 100
    return files._parse_fit_positions(positions)


def get_api(api=None):
    return api or _API or login()


def login(email=None, password=None, credentials=None, max_retries=5):
    creds = {
        'email': email,
        'password': password,
    }
    if credentials:
        with open(credentials, 'r') as f:
            data = json.load(f)
            data['email'] = data.pop("username")
            creds.update(data)

    if creds['email'] is None:
        print("please input your Garmin email address: ")
        creds['email'] = input()
    if creds['password'] is None:
        creds['password'] = getpass.getpass('Input your Garmin password: ')

    for i in range(max_retries):
        try:
            # API
            # Initialize Garmin api with your credentials
            api = Garmin(**creds)
            api.login()
            global _API
            _API = api
            break
        except (
            GarminConnectConnectionError,
            GarminConnectAuthenticationError,
            GarminConnectTooManyRequestsError,
        ) as err:
            logging.error(
                "Error occurred during Garmin Connect communication: %s", err)
            if i + 1 == max_retries:
                raise err

    return api


def download_activity(activity_id, path=None, api=None):
    api = get_api(api)

    gpx_data = api.download_activity(
        activity_id, dl_fmt=api.ActivityDownloadFormat.GPX)
    path = path or f"./{str(activity_id)}.gpx"
    with open(path, "wb") as fb:
        fb.write(gpx_data)

    return path


def _download_activities(activities, folder='./', max_workers=4, api=None, **kwargs):
    api = get_api(api)

    activity_ids = (act["activityId"] for act in activities)
    inputs = {
        act_id: (act_id, os.path.join(folder, f"{str(act_id)}.gpx"), api)
        for act_id in activity_ids
    }
    return map_concurrent(
        download_activity, inputs,
        threaded=True, max_workers=max_workers,
        raise_on_err=False, **kwargs
    )


def load_activity(activity_id, api=None):
    api = get_api(api)

    f = api.download_activity(
        activity_id, dl_fmt=api.ActivityDownloadFormat.GPX)
    return parse_gpx_data(gpxpy.parse(f))


def load_activities(activity_ids, max_workers=4, api=None, **kwargs):
    api = get_api(api)
    inputs = {
        act_id: (act_id, api) for act_id in activity_ids
    }
    return map_concurrent(
        load_activity, inputs,
        threaded=True, max_workers=max_workers,
        raise_on_err=False, **kwargs
    )


def download_fit(activity_id, path, api=None):
    api = get_api(api)

    zip_data = api.download_activity(
        activity_id, dl_fmt=api.ActivityDownloadFormat.ORIGINAL)
    with BytesIO(zip_data) as f, open(path, 'wb') as out:
        with zipfile.ZipFile(f, "r") as zipf:
            fit_file, = (
                f for f in zipf.filelist if f.filename.endswith("fit"))
            with zipf.open(fit_file, 'r') as f:
                shutil.copyfileobj(f, out)

    return path


def list_activity_fits(path, search="**/*[0-9].fit"):
    path = Path(path)
    fit_files = (
        (p, re.match(r"[0-9]+", p.name))
        for p in path.glob(search)
    )
    return {int(m.group()): p for p, m in fit_files if m}


def download_fits_to_folder(
    activity_ids,
    folder,
    activity_names=None,
    search="**/*[0-9].fit",
    max_workers=4,
    api=None
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    downloaded = list_activity_fits(folder, search=search)
    if activity_names is None:
        activity_names = activity_ids

    to_download = {}
    for activity_id, name in zip(activity_ids, activity_names):
        existing = downloaded.get(activity_id, None)
        path = folder / f"{name}.fit"
        if not existing and path != existing:
            path.parent.mkdir(exist_ok=True)
            to_download[activity_id] = (activity_id, path)

    errors = {}
    if to_download:
        downloaded, errors = map_concurrent(
            download_fit,
            to_download,
            max_workers=max_workers,
            api=get_api(api)
        )
    return downloaded, errors


def load_fit_activity(activity_id, api=None):
    api = get_api(api)

    zip_data = api.download_activity(
        activity_id, dl_fmt=api.ActivityDownloadFormat.ORIGINAL)
    return read_fit_zipfile(BytesIO(zip_data))


def load_fit_activities(activity_ids, max_workers=4, api=None, **kwargs):
    api = get_api(api)
    inputs = {
        act_id: (act_id, api) for act_id in activity_ids
    }
    return map_concurrent(
        load_fit_activity, inputs,
        threaded=True, max_workers=max_workers,
        raise_on_err=False, **kwargs
    )


def get_activities(
        start=0, limit=20, *, api=None, activityType=None, 
        startDate=None, endDate=None, minDistance=None, maxDistance=None, 
        **params
):
    if activityType:
        if activityType in _ACTIVITY_TYPES:
            params.update(_ACTIVITY_TYPES[activityType])
        else:
            params['activityType'] = activityType

    if startDate:
        if isinstance(startDate, datetime):
            startDate = startDate.strftime("%Y-%M-%d")
        params['startDate'] = startDate

    if endDate:
        if isinstance(endDate, datetime):
            endDate = endDate.strftime("%Y-%M-%d")
        params['endDate'] = endDate

    if minDistance:
        params['minDistance'] = str(minDistance)
    if maxDistance:
        params['maxDistance'] = str(maxDistance)

    return activities_to_dataframe(
        _get_activities(start=start, limit=limit, api=api, **params)
    )


def _get_activities(start=0, limit=20, *, api=None, **params):
    api = get_api(api)
    url = api.garmin_connect_activities
    params['start'] = start
    params['limit'] = limit

    print(params)

    return api.modern_rest_client.get(url, params=params).json()


def activities_to_dataframe(activities):
    df = pd.DataFrame.from_records(
        [dict(unflatten_json(act)) for act in activities]
    )
    depth = max(map(len, df.columns))
    df.columns = pd.MultiIndex.from_tuples([
        k + ('',) * (depth - len(k)) for k in df.columns
    ])
    return df


def download_and_process(activities, folder, save_file, api=None):
    fit_files, _ = download_activities(
        activities, folder, api=api
    )
    all_activities = process_fit_files(
        fit_files, save_file
    )
    return all_activities


def download_activities(activities, folder, api=None):
    activity_names = activities.startTimeLocal.str[:10].str.cat(
        activities.activityId.astype(str), sep="/"
    )
    return download_fits_to_folder(
        activities.activityId, folder, activity_names=activity_names, api=api
    )


def process_fit_files(
    fit_files,
    save_file,
):
    save_file = Path(save_file)

    if save_file.exists():
        all_activities = pd.read_parquet(save_file)
        processed = all_activities.index.get_level_values(0)
    else:
        all_activities = pd.DataFrame([])
        processed = set()

    to_load = {
        i: (str(fit_files[i]),)
        for i in fit_files.keys() - processed
    }
    if to_load:
        loaded, errors = map_concurrent(
            parse_fit_data, to_load, max_workers=4
        )
        if loaded:
            loaded = pd.concat(loaded, names=['activityId'])
            all_activities = pd.concat([all_activities, loaded])
            all_activities.to_parquet(save_file)

    return all_activities


def get_parser():
    def date(s):
        return datetime.strptime(s, '%Y-%m-%d')

    parser = argparse.ArgumentParser(
        description='Analyse recent gps data')
    parser.add_argument(
        'n',
        type=int, default=5, nargs='?',
        help='maximum number of activities to load')
    parser.add_argument(
        '--start', type=int, default=0, nargs='?',
        help="if loading large number of activities, sets when to "
        "start loading the activities from "
    )
    add_credentials_arguments(parser)
    parser.add_argument(
        '--actions',
        choices=['excel', "heartrate", 'download'],
        default=['excel', 'download'],
        nargs='+',
        help='specify action will happen'
    )
    parser.add_argument(
        '--excel-file',
        type=Path, default='garmin.xlsx', nargs='?',
        help='path of output excel spreadsheet'
    )
    parser.add_argument(
        '--folder',
        type=str, default='garmin_data', nargs='?',
        help='folder path to download fit files'
    )
    parser.add_argument(
        '-a', '--activity',
        type=str, nargs='?',
        help='activity type, options: rowing, cycling, running'
    )
    parser.add_argument(
        '--min-distance',
        type=int, nargs='?',
        help='minimum distance of activity (in km)'
    )
    parser.add_argument(
        '--max-distance',
        type=int, nargs='?',
        help='maximum distance of activity (in km)'
    )
    parser.add_argument(
        '--start-date',
        type=date,
        help='start date to search for activities from in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end-date',
        type=date,
        help='start date to search for activities from in YYYY-MM-DD format'
    )
    parser.add_argument(
        "--min-hr", type=int, default=60, nargs='?',
        help="min heart rate to plot"
    )
    parser.add_argument(
        "--max-hr", type=int, default=200, nargs='?',
        help="max heart rate to plot"
    )
    parser.add_argument(
        "--hr-to-plot",
        type=int,
        nargs='+',
        default=[60, 100, 120, 135, 142, 148, 155, 160, 165, 170, 175, 180],
        help="which heart rates to plot lines for"
    )
    parser.add_argument(
        "--cmap",
        choices=['gist_ncar', "inferno", 'hot', 'hot_r'],
        default="gist_ncar",
        help="The cmap to plot the heart rates for"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        '--hr-file', type=Path, default='heart_rate.xlsx',
        help="file to save heart rate to"
    )
    parser.add_argument(
        '--hr-plot', default='heart_rate.png',
        help="file to save heart rate to"
    )
    add_logging_argument(parser)
    return parser


def parse_args(args):
    return get_parser().parse_args(args)


def run(args=None):
    options = parse_args(args)
    set_logging(options)

    folder = Path(options.folder)
    save_file = Path(options.folder) / "activities.parquet"
    api = None
    additional_info = None
    all_activities = None

    if "download" in options.actions:
        api = login(options.user, options.password, options.credentials)

        additional_info = get_activities(
            options.start, options.start + options.n,
            activityType=options.activity,
            minDistance=options.min_distance,
            maxDistance=options.max_distance,
            startDate=options.start_date,
            endDate=options.end_date,
            api=api
        )
        additional_info.columns = [
            ".".join(str(i) for i in col if i != '')
            for col in additional_info.columns
        ]
        selected = additional_info.activityId
        all_activities = download_and_process(
            additional_info, folder, save_file, api=api)
        activities = all_activities.loc[selected]

    else:
        all_activities = pd.read_parquet(save_file)
        each_activity = all_activities.groupby(level=0)
        activity_info = pd.DataFrame({
            "startTime": each_activity.time.min(),
            "totalDistance": each_activity.distance.max(),
        }).sort_values("startTime", ascending=False)

        selected = activity_info
        if options.min_distance:
            activity_info = activity_info.loc[
                activity_info.totalDistance >= options.min_distance
            ]
        if options.max_distance:
            activity_info = activity_info.loc[
                activity_info.totalDistance <= options.max_distance
            ]
        if options.start_date:
            start = options.start_date
            # start = datetime.strptime(options.start_date, _YMD)

            activity_info = activity_info.loc[
                activity_info.startTime > start
            ]
        if options.end_date:
            end = options.end_date
            # end = datetime.strptime(options.end_date, _YMD)
            activity_info = activity_info.loc[
                activity_info.startTime < end + timedelta(days=1)
            ]

        selected = activity_info.index
        activities = all_activities.loc[selected]

    each_activity = activities.groupby(level=0)
    activity_info = pd.DataFrame({
        "startTime": each_activity.time.min(),
        "totalDistance": each_activity.distance.max(),
    }).sort_values("startTime", ascending=False)

    if 'excel' in options.actions:
        best_times, location_timings = activity_data_to_excel(
            activities,
            cols=['heart_rate', 'cadence', 'bearing'],
            additional_info=additional_info,
            xlpath=options.excel_file
        )
        print("best times: ")
        with pd.option_context('display.max_rows', None):
            print(best_times)

    if "heartrate" in options.actions:
        hrs = pd.RangeIndex(
            options.min_hr, options.max_hr, name='heart rate'
        )
        time_above_hr = splits.calc_time_above_hr(
            activities, hrs=hrs
        )
        time_above_hr.index = activity_info.startTime.loc[
            time_above_hr.index
        ]
        time_above_hr.sort_index(inplace=True)

        with pd.ExcelWriter(options.hr_file, mode='w') as xlf:
            time_above_hr.to_excel(
                xlf, "time above hr per session")
            time_above_hr.groupby(
                pd.Grouper(freq='w')
            ).sum().to_excel(xlf, "time above hr per week")
            time_above_hr.groupby(
                pd.Grouper(freq='M')
            ).sum().to_excel(xlf, "time above hr per month")

        print(f"saved heart rate data to {options.hr_file}")

        rolling_time_above_hr = time_above_hr.sort_index().rolling(
            pd.Timedelta(days=7),
            min_periods=1
        ).sum()
        reltime_above_hr = (
            rolling_time_above_hr / rolling_time_above_hr.values[:, [0]]
        )
        hr_to_plot = options.hr_to_plot
        cmap = 'gist_ncar'

        from .plot import plt, plot_heart_rates, mpl

        f, (ax1, ax2) = plt.subplots(2, figsize=(16, 16))

        _, _, hr_to_plot = plot_heart_rates(
            rolling_time_above_hr, hrs, hr_to_plot=hr_to_plot, cmap=cmap, ax=ax1)
        _, _, hr_to_plot = plot_heart_rates(
            reltime_above_hr * 100, hrs, hr_to_plot=hr_to_plot, cmap=cmap, ax=ax2)

        ax2.legend(
            bbox_to_anchor=(0., 1.02, 1., .102),
            ncol=len(hr_to_plot),
            loc=3,
            mode="expand",
            borderaxespad=0.
        )

        xlims = rolling_time_above_hr.index[0], rolling_time_above_hr.index[-1]
        ax1.set_xlim(*xlims)
        ax2.set_xlim(*xlims)
        ax1.xaxis.set_minor_locator(
            mpl.ticker.FixedLocator(np.arange(*ax1.get_xlim())))
        ax2.xaxis.set_minor_locator(
            mpl.ticker.FixedLocator(np.arange(*ax1.get_xlim())))
        ax2.set_xticklabels([])
        ax2.xaxis.set_ticks_position("top")
        ax2.set_xlabel("date")

        ax1.set_ylim(0, rolling_time_above_hr.max().max() * 1.05)
        ax1.set_ylabel("hours spent above HR bpm per week")
        ax2.set_ylim(100, 0)
        ax2.set_ylabel("relative time spent above HR bpm per week")
        ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())

        f.tight_layout()
        f.savefig(options.hr_plot, dpi=options.dpi)

        print(f"saved heart rate plot to {options.hr_plot}")

    return activities, additional_info


def main():
    try:
        run(sys.argv[1:])
    except Exception as err:
        logging.error(err)

    input("Press enter to finish")


if __name__ == "__main__":
    main()
