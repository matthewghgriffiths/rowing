
import datetime 
import time 
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .api import get_last_race_started
from .utils import (
    format_yaxis_splits, update_table, format_totalseconds, CURRENT_TIMEZONE
)
from .livetracker import get_current_data
from .predict import LivePrediction

class Dashboard:
    def __init__(
            self, race_tracker, 
            subplots_adjust=None, alpha=0.1, 
            table_bbox=(0, 1.05, 1, 0.6),
            **kwargs
    ):
        self.race_tracker = race_tracker
        
        self.alpha = alpha
        self.table_bbox = table_bbox
        self._subplots_adjust = subplots_adjust or {'hspace': 0.05, 'wspace': 0.05}
        
        kwargs.setdefault('figsize', (12, 8))
        self._init(**kwargs)

        self.finished = False
        self.p_pace = None
        self.p_times = None
        self.p_win = None
        self.p_dist = None
        self.p_finish = None

    @classmethod
    def from_race_id(cls, race_id, **kwargs):
        race_tracker = LivePrediction(race_id, noise=1)
        race_name = race_tracker.race_details.DisplayName
        race_start = pd.to_datetime(
            race_tracker.race_details.DateString
        ).tz_convert(CURRENT_TIMEZONE)
        print(f"loading {race_name}, starting at {race_start}")
        return cls(race_tracker, **kwargs)

    @classmethod 
    def load_last_race(cls, **kwargs):
        last_race = get_last_race_started()
        return cls.from_race_id(last_race.name, **kwargs)

    def live_dashboard(self):
        from IPython.display import display, clear_output
        for live_data in self.race_tracker.stream_livedata():
            clear_output(wait = True)
            if len(live_data):
                self.update(live_data)
                display(self.fig)
            else:
                print('no race data received')
                break

    def live_ion_dashboard(self):
        import matplotlib.pyplot as plt
        plt.ion()
        plt.show()
        for live_data in dash.race_tracker.stream_livedata():
            if len(live_data):
                current_data = get_current_data(live_data)
                print(current_data)
                dash.update(live_data)
                dash.fig.canvas.draw()
                dash.fig.canvas.flush_events()
            else:
                print('no race data received')
                break
        
    def _init(self, **kwargs):
        self._init_fig(**kwargs)
        self._init_tables()
        self._init_distance_behind()
        self._init_flags()
        self._init_pace()
        self._init_stroke_rate()
        self._init_predict()
        self._init_axes()
        
    def _init_fig(self, **kwargs):
        fig = plt.figure(**kwargs)
        gs = gridspec.GridSpec(ncols=6, nrows=3, figure=fig)
        left_axes = [fig.add_subplot(gs[0, 0])]
        middle_axes = [fig.add_subplot(gs[0, 1:5])]
        right_axes = [fig.add_subplot(gs[0, 5])]
        left_axes.extend([
            fig.add_subplot(gs[1, 0], sharex=left_axes[0]),
            fig.add_subplot(gs[2, 0], sharex=left_axes[0]),
        ])
        middle_axes.extend([
            fig.add_subplot(gs[1, 1:5], sharex=middle_axes[0]),
            fig.add_subplot(gs[2, 1:5], sharex=middle_axes[0]),
        ])
        right_axes.extend([
            fig.add_subplot(gs[1, 5], sharex=right_axes[0]),
            fig.add_subplot(gs[2, 5], sharex=right_axes[0]),
        ])
        all_axes = (left_axes, middle_axes, right_axes)
        for axes in all_axes:
            for ax in axes[1:]:
                ax.sharex(axes[0])
            for ax in axes[:-1]:
                ax.get_xaxis().set_visible(False)

        for axes in zip(*all_axes):
            for ax in axes[1:-1]:
                ax.sharey(axes[0])
                ax.get_yaxis().set_visible(False)
                
            axes[-1].yaxis.tick_right()
            axes[-1].yaxis.set_label_position("right")
        
        self.fig = fig
        self.left_axes, self.middle_axes, self.right_axes = all_axes
        
        self.axes = np.c_[
            self.left_axes, self.middle_axes, self.right_axes
        ]
        self.b_distance_behind_ax = self.left_axes[0]
        self.l_distance_behind_ax = self.middle_axes[0]
        self.p_distance_behind_ax = self.right_axes[0]
        self.p_distance_behind_ax.sharey(self.left_axes[0])
        
        self.b_pace_ax = self.left_axes[1]
        self.l_pace_ax = self.middle_axes[1]
        self.p_pace_ax = self.right_axes[1]
        self.p_pace_ax.get_shared_y_axes().remove(self.p_pace_ax)
        
        self.b_stroke_rates_ax = self.left_axes[-1]
        self.l_stroke_rates_ax = self.middle_axes[-1]
        
        self.p_win_ax = self.right_axes[-1]
        self.p_win_ax.get_shared_y_axes().remove(self.p_win_ax)

    def _init_tables(self):
        race_tracker = self.race_tracker
        bbox = self.table_bbox 
        n = len(race_tracker.lane_country)
        self.left_table = self.left_axes[0].table(
            np.full((n, 2), '-'),
            rowLabels=race_tracker.lane_country.values,
            rowColours=race_tracker.country_colors[
                race_tracker.lane_country
            ],
            colLabels=['lane', 'pos'],
            loc='top',
            bbox=bbox,
        )
        self.middle_table = self.middle_axes[0].table(
            np.full((n, 4), '-'),
            colLabels=[500, 1000, 1500, 2000],
            loc='top',
            bbox=bbox,
        )

        self.right_table = self.right_axes[0].table(
            np.full((n, 2), '-'), 
            colLabels=['PGMT', 'pred pos'],
            loc='top',
            bbox=bbox,
        )
        
    def _init_bar_line(self, bar_ax, line_ax):
        bars = self.race_tracker.bar(
            pd.Series(
                np.zeros(len(self.race_tracker.countries)),
                index=self.race_tracker.countries
            ),
            ax=bar_ax
        )
        lines = self.race_tracker.plot(
            [], 
            pd.DataFrame(
                np.zeros((0, len(self.race_tracker.countries))),
                columns=self.race_tracker.countries
            ), 
            ax=line_ax
        )
        return bars, lines
        
    def _init_distance_behind(self):
        self.b_distance_behind, self.l_distance_behind = self._init_bar_line(
            self.b_distance_behind_ax, self.l_distance_behind_ax
        )

    def _init_flags(self):
        self.b_distance_flags = self.race_tracker.plot_flags(
            self.race_tracker.country_lane * 0,
            zoom=0.015,
            zorder=10,
            ax=self.b_distance_behind_ax
        )
        
    def _init_pace(self):
        self.b_pace, self.l_pace = self._init_bar_line(
            self.b_pace_ax, self.l_pace_ax
        )
        
    def _init_stroke_rate(self):
        self.b_stroke_rates, self.l_stroke_rates = self._init_bar_line(
            self.b_stroke_rates_ax, self.l_stroke_rates_ax
        )
        
    def _init_predict(self):
        self.p_win = self.race_tracker.bar(
            pd.Series(
                np.zeros(len(self.race_tracker.countries)),
                index=self.race_tracker.countries
            ),
            ax=self.right_axes[-1]
        )
        
    def _init_axes(self):
        comp_name = self.race_tracker.competition_details.DisplayName
        race_name = self.race_tracker.race_details.DisplayName
        date = pd.to_datetime(
            self.race_tracker.race_details.DateString
        ).astimezone(CURRENT_TIMEZONE).strftime("%c (%Z)")
        progression = self.race_tracker.race_details.Progression
        if progression:
            progression = f"\nProgression: {progression}"
        else:
            progression = ''

        fig_title = f"{comp_name}: {race_name}, {date}{progression}"
        self.fig.suptitle(fig_title)

        self.b_distance_behind_ax.set_ylabel('distance from leader (m)')
        self.b_pace_ax.set_ylabel('split (/500m)')
        self.b_stroke_rates_ax.set_ylabel('rate (s/m)')
        
        self.p_distance_behind_ax.set_ylabel('predicted finish')
        self.p_pace_ax.set_ylabel('predicted time')
        self.p_win_ax.set_ylabel('probability of win')
        
        self.axes[-1, 0].set_xlabel('country')  
        self.axes[-1, 1].set_xlabel('distance (m)')
        
        self.axes[-1, 0].set_xticklabels(
            self.axes[-1, -1].get_xticklabels(), 
            rotation=-45,
        )
        self.axes[-1, -1].set_xticklabels(
            self.axes[-1, -1].get_xticklabels(), 
            rotation=45,
        )
        self.fig.tight_layout()
        self.fig.subplots_adjust(**self._subplots_adjust)

    def update(self, live_data=None, predictions=None):
        self.update_livedata(live_data)
        self.update_predictions(predictions=predictions, live_data=live_data)
        
    def update_livedata(self, live_data=None):
        if live_data is None:
            live_data = self.race_tracker.update_livedata()
            
        if len(live_data):
            self.update_stroke_rate(live_data)
            self.update_pace(live_data)
            self.update_distance_behind(live_data)
            self.update_pos(live_data)
            self.update_intermediates(
                self.race_tracker.intermediate_results)

    def update_pos(self, live_data):
        update_table(
            self.left_table, 
            self.race_tracker.lane_country, 
            ['lane', 'pos'], 
            pd.DataFrame({
                'lane': self.race_tracker.country_lane,
                'pos': live_data.currentPosition.iloc[-1].loc[
                    self.race_tracker.lane_country
                ]
            })
        )

    def update_intermediates(self, intermediates):
        update_table(
            self.middle_table, 
            self.race_tracker.lane_country, 
            [500, 1000, 1500, 2000], 
            intermediates.applymap(
                format_totalseconds, 
                na_action='ignore'
            ).fillna('-')
        )

        if 2000 in intermediates.columns:
            self.finished = True
            countries = self.race_tracker.lane_country
            finish_times = intermediates.reindex(countries)[2000]
            finish_pos = finish_times.sort_values()
            finish_pos[:] = np.arange(1, len(finish_pos) + 1)
            pgmt = (self.race_tracker.gmt / finish_times).apply("{:.1%}".format)
        
            update_table(
                self.right_table, 
                self.race_tracker.lane_country, 
                ['PGMT', 'pos'],
                pd.DataFrame({
                    'PGMT': pgmt[countries],
                    'pos': finish_pos.astype(int)[countries], 
                }) 
            )
            self.right_table[0, 1].get_text().set_text('final pos')
        
    def update_distance_behind(self, live_data):
        # self.race_tracker.update_bar(
        #     self.b_distance_behind, 
        #     live_data.distanceFromLeader.iloc[-1],
        # )        
        self.race_tracker.update_plot(
            self.l_distance_behind, 
            live_data.distanceTravelled,
            live_data.distanceFromLeader,
        )
        self.race_tracker.update_flags(
            self.b_distance_flags,
            live_data.distanceFromLeader.iloc[-1],
        )
        self.b_distance_behind_ax.set_ylim(
            live_data.distanceFromLeader.values.max() + 1, 
            -5, 
        )
        
        
    def update_pace(self, live_data):
        pace = 500 / live_data.metrePerSecond
        self.race_tracker.update_bar(
            self.b_pace, 
            pace.iloc[-1],
        )
        self.race_tracker.update_plot(
            self.l_pace, 
            live_data.distanceTravelled,
            pace,
        )            
        ylim = (pace.values[-100:].max() + 1, pace.values.min() - 1)
        self.b_pace_ax.set_ylim(ylim)
        format_yaxis_splits(self.b_pace_ax)
        self.b_pace_ax.set_ylim(ylim)
        
            
    def update_stroke_rate(self, live_data):
        self.race_tracker.update_bar(
            self.b_stroke_rates, 
            live_data.strokeRate.iloc[-1],
        )        
        self.race_tracker.update_plot(
            self.l_stroke_rates, 
            live_data.distanceTravelled,
            live_data.strokeRate,
        )
        
        self.b_stroke_rates_ax.set_ylim(
            25, live_data.strokeRate.values.max() + 1
        )
        
    def plot_predictions(self, predictions):
        preds_pace, preds_time, preds_dist, win_probs = predictions
        self.plot_pred_distance(*preds_dist)
        self.plot_pred_pace(*preds_pace)
        self.plot_finish_times(*preds_time)
        self.plot_win_probs(win_probs)
        
    def update_predictions(self, predictions=None, live_data=None):
        if predictions is None:
            predictions = self.race_tracker.predict(live_data=live_data)

        if predictions:
            preds_pace, preds_time, preds_dist, win_probs = predictions
            self.update_pred_distance(*preds_dist)
            self.update_pred_pace(*preds_pace)
            self.update_finish_times(*preds_time)
            self.update_win_probs(win_probs)
            
    def plot_pred_distance(self, pred_dist, pred_dist_std):
        pred_dist_behind = pred_dist.max(1).values[:, None] - pred_dist
        
        self.p_dist = \
            self.race_tracker.plot_uncertainty(
                pred_dist_behind, pred_dist_std,
                ax=self.middle_axes[0], alpha=self.alpha
            )
        self.p_finish = \
            self.race_tracker.plot_finish(
                pred_dist_behind.iloc[-1],
                pred_dist_std.iloc[-1],
                set_lims=False,
                ax=self.right_axes[0], 
            )

        if self.race_tracker.completed:
            ymax = pred_dist_behind.values.max()  +5
        else:
            ymax = pred_dist_behind.iloc[-1].max() + 5
        self.right_axes[0].set_ylim(ymax, -9)
        
    def update_pred_distance(self, pred_dist, pred_dist_std):
        if self.p_dist is None:
            self.plot_pred_distance(pred_dist, pred_dist_std)
        else:
            pred_dist_behind = pred_dist.max(1).values[:, None] - pred_dist
            self.race_tracker.update_plot_uncertainty(
                *self.p_dist,
                pred_dist_behind, pred_dist_std,
            )
            self.race_tracker.update_plot_finish(
                *self.p_finish,
                pred_dist_behind.iloc[-1],
                pred_dist_std.iloc[-1],
            )
            
            if self.race_tracker.completed:
                ymax = pred_dist_behind.values.max()  + 5
            else:
                ymax = pred_dist_behind.iloc[-1].max() + 5
            self.right_axes[0].set_ylim(ymax, -9)
        
    def plot_pred_pace(self, pred_pace, pred_pace_std):
        self.p_pace = \
            self.race_tracker.plot_uncertainty(
                pred_pace, pred_pace_std, 
                ax=self.middle_axes[1], alpha=self.alpha
            )
        
    def update_pred_pace(self, pred_pace, pred_pace_std):
        if self.p_pace is None:
            self.plot_pred_pace(pred_pace, pred_pace_std)
        else:
            self.race_tracker.update_plot_uncertainty(
                *self.p_pace, pred_pace, pred_pace_std, 
            )
        
    def plot_finish_times(self, pred_times, pred_times_std):
        ax = self.right_axes[1]
        self.p_times = \
            self.race_tracker.plot_finish(
                pred_times.iloc[-1], pred_times_std.iloc[-1], 
                ax=ax
            )
        ylims = ax.get_ylim()
        format_yaxis_splits(ax)
        ax.set_ylim(*ylims[::-1])   

        if not self.finished:
            countries = self.race_tracker.lane_country
            finish_times = pred_times.iloc[-1]
            finish_pos = finish_times.sort_values()
            finish_pos[:] = np.arange(1, len(finish_pos) + 1)
            pgmt = (self.race_tracker.gmt / finish_times).apply("{:.1%}".format)
            logging.info(pgmt[countries])
            logging.info(finish_pos[countries].astype(int))
            update_table(
                self.right_table, 
                countries, 
                ['PGMT', 'pos'],
                pd.DataFrame({
                    'PGMT': pgmt[countries],
                    'pos': finish_pos[countries].astype(int), 
                }) 
            )
        
    def update_finish_times(self, pred_times, pred_times_std):
        if self.finished:
            return None 

        if self.p_times is None:
            self.plot_finish_times(pred_times, pred_times_std)
        else:
            from matplotlib.ticker import AutoLocator
            ylims = self.race_tracker.update_plot_finish(
                *self.p_times, pred_times.iloc[-1], pred_times_std.iloc[-1], 
            )
            self.right_axes[1].yaxis.set_major_locator(AutoLocator()) 
            format_yaxis_splits(self.right_axes[1])    
            self.right_axes[1].set_ylim(*ylims[::-1])   
        
    def plot_win_probs(self, win_probs):
        self.p_win = self.race_tracker.bar(
            win_probs, ax=self.p_win_ax
        )
        self.p_win_ax.set_ylim(0, np.nan_to_num(np.nanmax(win_probs)) + 0.05)
        
    def update_win_probs(self, win_probs):
        if self.p_win is None:
            self.plot_win_probs(win_probs)
        else:
            self.race_tracker.update_bar(self.p_win, win_probs)
            self.p_win_ax.set_ylim(0, np.nan_to_num(np.nanmax(win_probs)) + 0.05)


def main():
    import matplotlib.pyplot as plt
    from world_rowing.api import show_next_races

    dash = Dashboard.load_last_race(figsize=(12, 8))
    dash.live_ion_dashboard()
        
    print('race results:')
    print(dash.race_tracker.intermediate_results)

    print('\nupcoming races:')
    print(show_next_races())
    plt.show(block=True)


if __name__ == "__main__":
    main()