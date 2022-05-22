
import sys
import argparse
import zipfile
from pathlib import Path
import logging

import gpxpy
import fitparse
import pandas as pd
import numpy as np

from . import geodesy, utils, splits

_SEMICIRCLE_SCALE = 180 / 2**31

def read_gpx(filename):
    with open(filename, 'r') as f:
        gpx_data = gpxpy.parse(f)

    return parse_gpx_data(gpx_data)

def parse_gpx_data(gpx_data):
    positions = pd.DataFrame.from_records(
        {
            'latitude': point.latitude, 
            'longitude': point.longitude, 
            'time': point.time
        } 
        for track in gpx_data.tracks 
        for segment in track.segments 
        for point in segment.points
    )
    if positions.empty:
        return pd.DataFrame(
            [], columns=[
                'latitude', 'longitude', 'time', 'timeElapsed', 
                'distanceDelta', 'distance', 'bearing_r', 'bearing'
        ]) 

    last = positions.index[-1]
    positions['timeElapsed'] = positions.time - positions.time[0]
    positions['distanceDelta'] = geodesy.haversine_km(positions, positions.shift(-1))
    positions.loc[last, 'distanceDelta'] = 0
    positions['distance'] = np.cumsum(positions.distanceDelta)
    positions['bearing_r'] = geodesy.rad_bearing(positions, positions.shift(-1))
    positions.loc[0, 'bearing_r'] = positions.bearing_r[1]
    positions['bearing'] = np.rad2deg(positions.bearing_r)

    return positions


def read_fit_zipfile(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        fit_file, = (f for f in zipf.filelist if f.filename.endswith("fit"))
        return read_fit_file(fit_file, zipf.open, mode='r')


def read_fit_file(fit_file, open=open, mode='rb'):
    with open(fit_file, mode) as f:
        return parse_fit_data(f)


def peak_fit_file(fit_file):
    fit_data = fitparse.FitFile(fit_file)
    record = next(fit_data.get_messages("record"))
    return {f.name: f.value for f in record.fields}


def parse_fit_data(fit_file):
    fit_data = fitparse.FitFile(fit_file)
    positions = pd.DataFrame.from_records(
        {f.name: f.value for f in record.fields}
        for record in fit_data.get_messages("record")
    ).rename(columns={'timestamp': 'time'})
    if 'position_lat' in positions.columns:
        positions = positions.dropna(
            subset=['position_lat', 'position_long']
        ).reset_index(drop=True)

    last = positions.index[-1]
    positions['distance'] /= 1000
    positions['timeElapsed'] = positions.time - positions.time.iloc[0]
    positions['timeDelta'] = - positions.time.diff(-1)
    positions.loc[last, 'timeDelta'] = pd.Timedelta(0)

    if 'position_lat' in positions.columns:
        positions['latitude'] = positions.position_lat * _SEMICIRCLE_SCALE
        positions['longitude'] = positions.position_long * _SEMICIRCLE_SCALE
        positions['distanceDelta'] = - positions.distance.diff(-1)
        positions.loc[last, 'distanceDelta'] = 0
        positions['bearing_r'] = geodesy.rad_bearing(positions, positions.shift(-1))
        positions.loc[last, 'bearing_r'] = positions.bearing_r.iloc[-2]
        positions['bearing'] = np.rad2deg(positions.bearing_r)

    return positions


def activity_data_to_excel(
        activities, 
        locations=None, cols=None,
        additional_info=None, sheet_names=None, 
        xlpath='excel_data.xlsx',
):
    activity_info, best_times, location_timings = splits.process_activities(
        activities, locations=locations, cols=cols
    )
    if additional_info is not None:
        activity_info.join(additional_info)
    if sheet_names is None:
        sheet_names = pd.Series(
            activity_info.startTime.dt.strftime("%Y-%m-%d").str.cat(
                activity_info.index.astype(str), sep=' '
            )
        )
    
    with pd.ExcelWriter(xlpath) as xlf:
        activity_info.to_excel(xlf, "activities")
        best_times.loc[:, ['time', 'split']] = best_times[['time', 'split']].applymap(
            utils.strfsplit)
        best_times.to_excel(xlf, "best_times")
        for actid, timings in location_timings.items():
            if not timings.empty:
                timings.applymap(utils.strfsplit).to_excel(
                    xlf, sheet_names.loc[actid])

def get_parser():
    parser = argparse.ArgumentParser(
        description='Analyse gpx data files')
    
    parser.add_argument(
        "gpx_file", type=str, nargs='*',
        help="gpx files to process, accepts globs, e.g. activity_*.gpx"
        ", default='*.gpx'"
    )
    parser.add_argument(
        '-o', '--out-file', type=str, nargs="?",
        default='gpx_data.xlsx',
        help="path to excel spreadsheet to save results to"
        ", default='gpx_data.xlsx'"
    )
    utils.add_logging_argument(parser)
    
    return parser


def run(args=None):
    path = Path()
    options = get_parser().parse_args(args)
    utils.set_logging(options)

    gpx_files = {
        f.name: (f,) for gpx_file in options.gpx_file or ['*.gpx']
        for f in path.glob(gpx_file)
    }
    print("processing %d gpx files" % len(gpx_files))
    activity_data, errors = utils.map_concurrent(read_gpx, gpx_files)
    
    if not activity_data:
        print("no gpx files could be parsed")
        return 
        
    for data in activity_data.values():
        data.time = pd.to_datetime(data.time).dt.tz_localize(None)
    
    activities = pd.DataFrame.from_dict({
        k: data.iloc[-1] for k, data in activity_data.items()
        }, 
        orient='index'
    ).rename(columns={
        "distance": "totalDistance",
        "time": "endTime"
    })
    activities.index.name = 'filename'
    for k, data in activity_data.items():
        activities.loc[k, 'startTime'] = data.time.iloc[0]

    activities.sort_values(
        by='startTime', ascending=False, inplace=True
    )
    print("saving analysis to %s" % options.out_file)
    activity_data_to_excel(
        activities, activity_data, locations=None, 
        xlpath=options.out_file,
    )

def main():
    try:
        run(sys.argv[1:])
    except Exception as err:
        logging.error(err) 
        
    input("Press enter to finish")

if __name__ == "__main__":
    main()