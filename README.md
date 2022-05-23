# rowing.world_rowing
Collection of code to load, process and analyse rowing data


# rowing.analysis
A python library for analysing gps data

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
usage: garmin [-h] [--start [START]] [-u [USER]] [-p [PASSWORD]] [-c [CREDENTIALS]] [--actions {excel,heartrate,download} [{excel,heartrate,download} ...]] [--excel-file [EXCEL_FILE]] [--folder [FOLDER]] [-a [ACTIVITY]] [--min-distance [MIN_DISTANCE]]
                 [--max-distance [MAX_DISTANCE]] [--start-date START_DATE] [--end-date END_DATE] [--min-hr [MIN_HR]] [--max-hr [MAX_HR]] [--hr-file HR_FILE] [--hr-plot HR_PLOT] [--hr-to-plot HR_TO_PLOT [HR_TO_PLOT ...]] [--cmap {gist_ncar,inferno,hot,hot_r}] [--dpi DPI]
                 [-l-log LOG]
                 [n]

Analyse recent gps data

positional arguments:
  n                     maximum number of activities to load

optional arguments:
  -h, --help            show this help message and exit
  --start [START]       if loading large number of activities, sets when to start loading the activities from
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
  --hr-file HR_FILE     file to save heart rate to
  --hr-plot HR_PLOT     file to save heart rate to
  --hr-to-plot HR_TO_PLOT [HR_TO_PLOT ...]
                        which heart rates to plot lines for
  --cmap {gist_ncar,inferno,hot,hot_r}
                        The cmap to plot the heart rates for, options
  --dpi DPI
  -l-log LOG, --log LOG, --logging LOG
                        Provide logging level. Example --log debug', default='warning'
```
