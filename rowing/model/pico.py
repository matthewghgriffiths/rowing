
import json 

import pandas as pd 

from ..analysis import geodesy

accel_cols = ['accX', 'accY', 'accZ']
gyro_cols = ['gyroX', 'gyroY', 'gyroZ']

def load_raw_pico_data(json_path):
    with open(json_path, 'rb') as f:
        capture = json.load(f)

    data_groups = {}
    for elem in capture['log_entries']:
        data_groups.setdefault(
            elem['Name'], {}
        )[elem['Timestamp']] = elem['Fields']

    capture_data = {
        k: pd.DataFrame.from_dict(group, orient='index')
        for k, group in data_groups.items()
    }
    return capture_data

def process_pico_data(capture_data):
    accelgyro_data = capture_data['AccelGyro'].sort_index()
    accelgyro_data.index.name = 'timestamp'
    accelgyro_data.index /= 1e6

    gps_data = capture_data['UBX0'].sort_index()

    gps_data.index.name = 'timestamp'
    gps_data['latitude'] = gps_data.Lat
    gps_data['longitude'] = gps_data.Lon 

    valid_gps_data = gps_data.loc[
        (gps_data.Hor_accuracy < 1000)
        & (gps_data.HDOP > 0)
        & (gps_data.HDOP < 5)
    ].copy()
    valid_gps_data.index /= 1e6
    valid_gps_data['distanceDelta'] = geodesy.haversine_km(valid_gps_data, valid_gps_data.shift()).fillna(0)
    valid_gps_data['distance'] = valid_gps_data.distanceDelta.cumsum()
    valid_gps_data['timeElapsed'] = pd.to_timedelta(valid_gps_data.index, 's')

    return accelgyro_data, valid_gps_data