
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
from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError,
)

from .utils import (
    map_concurrent, unflatten_json, strfsplit, 
    add_logging_argument, set_logging, _YMD
)
from .files import parse_gpx_data, read_fit_zipfile, parse_fit_data, activity_data_to_excel
from . import splits


logger = logging.getLogger(__name__)

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

def get_api(api=None):
    return api or _API or login()

def login(email=None, password=None, credentials=None, max_retries=5):
    creds = {
        'email': email,
        'password': password,
    }
    if credentials:
        with open(credentials, 'r') as f:
            creds.update(json.load(f))
            

    if creds['email'] is None:
        print("please input your Garmin email address: ")
        creds['email'] = input()
    if creds['password'] is None:
        creds['password'] = getpass.getpass('Input your Garmin password: ')

    for i in range(max_retries):
        try:
            # API
            ## Initialize Garmin api with your credentials
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
            logging.error("Error occurred during Garmin Connect communication: %s", err)
            if i + 1 == max_retries:
                raise err

    return api


def download_activity(activity_id, path=None, api=None):
    api = get_api(api)

    gpx_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.GPX)
    path = path or f"./{str(activity_id)}.gpx"
    with open(path, "wb") as fb:
        fb.write(gpx_data)

    return path


def download_activities(activities, folder='./', max_workers=4, api=None, show_progress=True):
    api = get_api(api)

    activity_ids = (act["activityId"] for act in activities)
    inputs = {
        act_id: (act_id, os.path.join(folder, f"{str(act_id)}.gpx"), api)
        for act_id in activity_ids
    }
    return map_concurrent(
        download_activity, inputs, 
        threaded=True, max_workers=max_workers, 
        show_progress=show_progress, raise_on_err=False
    )

def load_activity(activity_id, api=None):
    api = get_api(api)

    f = api.download_activity(
        activity_id, dl_fmt=api.ActivityDownloadFormat.GPX)
    return parse_gpx_data(gpxpy.parse(f))

def load_activities(activity_ids, max_workers=4, api=None, show_progress=True):
    api = get_api(api)
    inputs = {
        act_id: (act_id, api) for act_id in activity_ids
    }
    return map_concurrent(
        load_activity, inputs, 
        threaded=True, max_workers=max_workers, 
        show_progress=show_progress, raise_on_err=False
    )

def download_fit(activity_id, path, api=None):
    api = get_api(api)

    zip_data = api.download_activity(
        activity_id, dl_fmt=api.ActivityDownloadFormat.ORIGINAL)
    with BytesIO(zip_data) as f, open(path, 'wb') as out:
        with zipfile.ZipFile(f, "r") as zipf:
            fit_file, = (f for f in zipf.filelist if f.filename.endswith("fit"))
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
        activity_names = None,
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

def load_fit_activities(activity_ids, max_workers=4, api=None, show_progress=True):
    api = get_api(api)
    inputs = {
        act_id: (act_id, api) for act_id in activity_ids
    }
    return map_concurrent(
        load_fit_activity, inputs, 
        threaded=True, max_workers=max_workers, 
        show_progress=show_progress, raise_on_err=False
    )

def get_activities(start=0, limit=20, *, api=None, activityType=None, startDate=None, endDate=None, minDistance=None, maxDistance=None, **params):
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
    
    if minDistance: params['minDistance'] = str(minDistance)
    if maxDistance: params['maxDistance'] = str(maxDistance)

    return activities_to_dataframe(
        _get_activities(start=start, limit=limit, api=api, **params)
    )


def _get_activities(start=0, limit=20, *, api=None, **params):
    api = get_api(api)
    url = api.garmin_connect_activities
    params['start'] = start 
    params['limit'] = limit 

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
        loaded, errors =  map_concurrent(
            parse_fit_data, to_load, max_workers=4
        )
        if loaded:
            loaded = pd.concat(loaded, names = ['activityId'])
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
    parser.add_argument(
        '-u', '--user', '--email',
        type=str, nargs='?',
        help='Email address to use'
    )
    parser.add_argument(
        '-p', '--password',
        type=str, nargs='?',
        help='Password'
    )
    parser.add_argument(
        '-c', '--credentials',
        type=str, nargs='?',
        help='path to json file containing credentials (email and password)'
    )
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
        default = [60, 100, 120, 135, 142, 148, 155, 160, 165, 170, 175, 180],
        help="which heart rates to plot lines for"
    )
    parser.add_argument(
        "--cmap",
        choices=['gist_ncar', "inferno", 'hot', 'hot_r'],
        default = "gist_ncar",
        help="The cmap to plot the heart rates for"
    )
    parser.add_argument(
        "--dpi",
        type=int, 
        default = 300,
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
        activity_data_to_excel(
            activities,
            cols=['heart_rate', 'cadence', 'bearing'],
            additional_info=additional_info,
            xlpath=options.excel_file
        )

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
                xlf, f"time above hr per session")
            time_above_hr.groupby(
                pd.Grouper(freq='w')
            ).sum().to_excel(xlf, f"time above hr per week")
            time_above_hr.groupby(
                pd.Grouper(freq='M')
            ).sum().to_excel(xlf, f"time above hr per month")

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
        ax1.xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(*ax1.get_xlim())))
        ax2.xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(*ax1.get_xlim())))
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


    return activities, additional_info

def main():
    try:
        run(sys.argv[1:])
    except Exception as err:
        logging.error(err) 
        
    input("Press enter to finish")

if __name__ == "__main__":
    main()