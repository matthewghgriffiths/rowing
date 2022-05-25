# world_rowing
Collection of code to load, process and analyse rowing data from [World Rowing](https://worldrowing.com/).

To explore data from world rowing, run `world_rowing`, for example `pgmts` allows you to explore the pgmts for different boat classes/events/finals.
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

It is possible to view livetrack data from world rowing by calling `view` or `view_race`. 

```
rowing> view
   1. 2022 World Rowing Cup I
   2. 2022 World Rowing Cup II
   3. 2022 World Rowing Cup III
   4. 2022 World Rowing Under 23 Championships
   5. 2022 World Rowing Under 19 Championships
   6. 2022 European Rowing Championships
   7. 2022 World Rowing Championships
Select which competition you want: 
```
Competition, event and race selection are achieved by entering the appropriate number.

It is possible to pass the competition, event and race numbers directly to the `view` command to avoid having to entering them one by one, 
```
rowing> view 2021 8 9
selecting option 8. 2020 Olympic Games Regatta
selecting option 9. Men's Eight
   1. Men's Eight Heat 1
   2. Men's Eight Heat 2
   3. Men's Eight Repechage R1
   4. Men's Eight Final FA
Select which race you want: 4
```
Running this command will show the following visualisation of the race,
![View of livetracker data from race](/race.png)

Running `livetracker` in the `world_rowing` while a race is running will show a graph of the livetracker data updated in real time, like below, 
```
rowing> livetracker
```
![animation of livetracker](/livetracker.gif)

This livetracker can be directly accessed by running `rowing_live_tracker`.

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