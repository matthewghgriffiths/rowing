
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .utils import format_yaxis_splits
from .predict import LivePrediction

class Dashboard:
    def __init__(self, race_tracker, subplots_adjust=None, alpha=0.1, **kwargs):
        self.race_tracker = race_tracker
        
        self.alpha = alpha
        self._subplots_adjust = subplots_adjust or {'hspace': 0.05, 'wspace': 0.05}
        
        
        kwargs.setdefault('figsize', (12, 8))
        self._init(**kwargs)

    @classmethod
    def from_race_id(cls, race_id, **kwargs):
        race_tracker = LivePrediction(race_id)
        return cls(race_tracker, **kwargs)
        
    def _init(self, **kwargs):
        self._init_fig(**kwargs)
        self._init_distance_behind()
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
        
        self.fig.subplots_adjust(**self._subplots_adjust)
        
    def update_bar(self, bars, heights, bottom=None):
        return self.race_tracker.update_bar(
            bars, heights, bottom=bottom
        )
        
    def update_livedata(self, live_data=None):
        if live_data is None:
            live_data = self.race_tracker.update_livedata()
            
        self.update_stroke_rate(live_data)
        self.update_pace(live_data)
        self.update_distance_behind(live_data)
        
        
    def update_distance_behind(self, live_data):
        self.race_tracker.update_bar(
            self.b_distance_behind, 
            live_data.distanceFromLeader.iloc[-1],
        )        
        self.race_tracker.update_plot(
            self.l_distance_behind, 
            live_data.distanceTravelled,
            live_data.distanceFromLeader,
        )
        self.b_distance_behind_ax.set_ylim(
            live_data.distanceFromLeader.values.max() + 1, 
            0, 
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
        self.b_pace_ax.set_ylim(
            np.nanmax([
                pace.values.max(),
                pace.values[50:].max()
            ]), 
            pace.values.min()
        )
        format_yaxis_splits(self.b_pace_ax)
        
            
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
        
    def update_predictions(self, predictions):
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
        self.right_axes[0].set_ylim(
            pred_dist_behind.iloc[-1].max() + 5, -5 
        )
        
    def update_pred_distance(self, pred_dist, pred_dist_std):
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
        self.right_axes[0].set_ylim(
            pred_dist_behind.iloc[-1].max() + 5, -5 
        )
        
    def plot_pred_pace(self, pred_pace, pred_pace_std):
        self.p_pace = \
            self.race_tracker.plot_uncertainty(
                pred_pace, pred_pace_std, 
                ax=self.middle_axes[1], alpha=self.alpha
            )
        
    def update_pred_pace(self, pred_pace, pred_pace_std):
        self.race_tracker.update_plot_uncertainty(
            *self.p_pace, pred_pace, pred_pace_std, 
        )
        
    def plot_finish_times(self, pred_times, pred_times_std):
        self.p_times = \
            self.race_tracker.plot_finish(
                pred_times.iloc[-1], pred_times_std.iloc[-1], 
                ax=self.right_axes[1]
            )
        ylims = self.right_axes[1].get_ylim()
        format_yaxis_splits(self.right_axes[1])
        self.right_axes[1].set_ylim(*ylims[::-1])
        
    def update_finish_times(self, pred_times, pred_times_std):
        ylims = self.race_tracker.update_plot_finish(
            *self.p_times, pred_times.iloc[-1], pred_times_std.iloc[-1], 
        )
        self.right_axes[1].set_ylim(*ylims[::-1])        
        
    def plot_win_probs(self, win_probs):
        self.p_win = self.race_tracker.bar(
            win_probs, ax=self.p_win_ax
        )
        self.p_win_ax.set_ylim(0, win_probs.max()*1.05)
        
    def update_win_probs(self, win_probs):
        self.race_tracker.update_bar(self.p_win, win_probs)
        self.p_win_ax.set_ylim(0, win_probs.max()*1.05)