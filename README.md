# world_rowing
Collection of code to load, process and analyse rowing data from [World Rowing](https://worldrowing.com/).

Can view the world rowing data by running `streamlit run world_rowing_app/home.py` or visiting https://matthewghgriffiths-worldrowing.streamlit.app/

# rowing.analysis
A python library for analysing gps data, there are two main programs, `gpx` and `garmin`. `gpx` directly processes gpx files, calculating fastest times/splits over distances and timings/splits between specified rowing landmarks. See `Garmin.ipynb` for a more direct example of how to use the library.

## Example usage
```
$ gpx --help
usage: gpx [-h] [-o [OUT_FILE]] [-l-log LOG] [gpx_file [gpx_file ...]]

Analyse gpx data files

positional arguments:
  gpx_file              gpx files to process, accepts globs, e.g. activity_*.gpx, default='*.gpx'

optional arguments:
  -h, --help            show this help message and exit
  -o [OUT_FILE], --out-file [OUT_FILE]
                        path to excel spreadsheet to save results, default='gpx_data.xlsx'
  -l-log LOG, --log LOG
                        Provide logging level. Example --log debug', default='warning'

$ garmin --help
usage: garmin    [-h] [--start [START]] [-u [USER]] [-p [PASSWORD]] [-c [CREDENTIALS]]
                 [--actions {excel,heartrate,download} [{excel,heartrate,download} ...]]
                 [--excel-file [EXCEL_FILE]] [--folder [FOLDER]] [-a [ACTIVITY]]
                 [--min-distance [MIN_DISTANCE]] [--max-distance [MAX_DISTANCE]]
                 [--start-date START_DATE] [--end-date END_DATE] [--min-hr [MIN_HR]]
                 [--max-hr [MAX_HR]] [--hr-to-plot HR_TO_PLOT [HR_TO_PLOT ...]]
                 [--cmap {gist_ncar,inferno,hot,hot_r}] [--dpi DPI] [--hr-file HR_FILE]
                 [--hr-plot HR_PLOT] [-l-log LOG]
                 [n]

Analyse recent gps data

positional arguments:
  n                     maximum number of activities to load

optional arguments:
  -h, --help            show this help message and exit
  --start [START]       if loading large number of activities, sets when to start
                        loading the activities from
  -u [USER], --user [USER], --email [USER]
                        Email address to use
  -p [PASSWORD], --password [PASSWORD]
                        Password
  -c [CREDENTIALS], --credentials [CREDENTIALS]
                        path to json file containing credentials (email and password)
  --actions {excel,heartrate,download} [{excel,heartrate,download} ...]
                        specify action will happen
  --excel-file [EXCEL_FILE]
                        path of output excel spreadsheet
  --folder [FOLDER]     folder path to download fit files
  -a [ACTIVITY], --activity [ACTIVITY]
                        activity type, options: rowing, cycling, running
  --min-distance [MIN_DISTANCE]
                        minimum distance of activity (in km)
  --max-distance [MAX_DISTANCE]
                        maximum distance of activity (in km)
  --start-date START_DATE
                        start date to search for activities from in YYYY-MM-DD format
  --end-date END_DATE   start date to search for activities from in YYYY-MM-DD format
  --min-hr [MIN_HR]     min heart rate to plot
  --max-hr [MAX_HR]     max heart rate to plot
  --hr-to-plot HR_TO_PLOT [HR_TO_PLOT ...]
                        which heart rates to plot lines for
  --cmap {gist_ncar,inferno,hot,hot_r}
                        The cmap to plot the heart rates for
  --dpi DPI
  --hr-file HR_FILE     file to save heart rate to
  --hr-plot HR_PLOT     file to save heart rate to
  -l-log LOG, --log LOG, --logging LOG
                        Provide logging level. Example --log debug', default='warning'
```

Example running `garmin`,

```
$ garmin --credentials garmin-credentials.json
best times: 
                                                                   time    split heart_rate cadence bearing
activity_id startTime           totalDistance length distance                                              
8864358195  2022-05-21 09:45:21 11.98812      250m   0.191      1:25.50  2:51.01       97.9     8.1    76.3
                                                     0.452      1:20.04  2:40.08      107.7    20.1    48.5
                                                     0.803      1:24.81  2:49.62      106.8    18.4    29.0
                                                     1.054      1:13.89  2:27.78      118.7    17.9    19.6
                                                     1.376      1:15.38  2:30.77      121.1    18.1    16.0
...                                                                 ...      ...        ...     ...     ...
8888463424  2022-05-25 05:35:52 16.04431      5km    2.943     22:35.28  2:15.52      153.4    20.0    11.4
                                                     8.142     25:10.40  2:31.04      151.9    20.1  -136.1
                                              7km    1.064     32:48.65  2:20.61      145.3    18.4    13.7
                                                     8.129     38:47.62  2:46.25      150.9    19.9  -141.6
                                              10km   1.757     47:57.81  2:23.89      150.0    19.7   -41.3
```

```
$ garmin --start-date 2021-09-01 --end-date 2022-05-25 --action heartrate --hr-plot hr.png
saved heart rate data to heart_rate.xlsx
saved heart rate plot to hr.png
Press enter to finish
```
`hr.png` is shown below, 
 
![heart rate plot](/hr.png)