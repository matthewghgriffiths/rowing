# rowing.world_rowing
Collection of code to load, process and analyse rowing data

```
$ world_rowing
Welcome try running `pgmts`, `race` or `livetracker`
rowing> pgmts 2021
   1. 2021 World Rowing European Olympic and Paralympic Qualification Regatta
   2. 2021 European Rowing Championships
   3. 2021 World Rowing Cup I
   4. 2021 World Rowing Final Olympic Qualification Regatta
   5. 2021 World Rowing Cup II
   6. 2021 World Rowing Cup III
   7. 2021 World Rowing Under 23 Championships
   8. 2020 Olympic Games Regatta
   9. 2021 World Rowing Junior Championships
  10. 2020 Paralympic Games Regatta
  11. 2021 European Rowing Under 23 Championships
Select which competition you want 8
loaded PGMTS for 687 results
   1. by result
   2. by boat class
   3. by final
   4. plot by boat class
How to display PGMTs?2
         PGMT     Time      WBT Country  Rank  Lane                Date
Boat                                                                   
LM2x  100.00%  6:05.33  6:05.33     IRL     1     3 2021-07-28 02:30:00
LW2x  100.00%  6:41.36  6:41.36     ITA     1     4 2021-07-28 02:50:00
M4x   100.00%  5:32.03  5:32.03     NED     1     4 2021-07-28 01:30:00
W2-   100.00%  6:47.41  6:47.41     NZL     1     4 2021-07-28 03:30:00
W4x   100.00%  6:05.13  6:05.13     CHN     1     3 2021-07-28 01:50:00
W8+   100.00%  5:52.99  5:52.99     ROU     1     4 2021-07-28 03:40:00
M2x    99.83%  6:00.33  5:59.72     FRA     1     3 2021-07-28 00:30:00
W4-    99.73%  6:15.37  6:14.36     AUS     1     3 2021-07-28 00:50:00
W2x    99.07%  6:41.03  6:37.31     ROU     1     4 2021-07-28 00:18:00
M8+    98.96%  5:22.04  5:18.68     NZL     1     3 2021-07-28 03:50:00
M2-    98.66%  6:13.51  6:08.50     ROU     1     4 2021-07-28 03:00:00
M4-    98.57%  5:42.76  5:37.86     AUS     1     3 2021-07-28 01:10:00
W1x    98.56%  7:13.97  7:07.71     NZL     1     4 2021-07-30 00:33:00
M1x    97.58%  6:40.45  6:30.74     GRE     1     4 2021-07-30 00:45:00
```

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
