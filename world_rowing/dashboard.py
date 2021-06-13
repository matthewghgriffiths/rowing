
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
from .livetracker import get_current_data, RaceTracker
from .predict import LivePrediction

logger = logging.getLogger('world_rowing.dashboard')


class Dashboard:
    def __init__(
            self, race_tracker: RaceTracker,
            subplots_adjust=None, alpha=0.1,
            table_bbox=(0, 1.05, 1, 0.6),
            **kwargs
    ):
        self.race_tracker = race_tracker

        self.alpha = alpha
        self.table_bbox = table_bbox
        self._subplots_adjust = subplots_adjust or {
            'hspace': 0.05, 'wspace': 0.05}

        kwargs.setdefault('figsize', (12, 8))
        self._init(**kwargs)

        self.finished = False
        self.p_pace = None
        self.p_times = None
        self.p_win = None
        self.p_behind = None
        self.p_finish = None

    @classmethod
    def from_race_id(cls, race_id, **kwargs):
        race_tracker = LivePrediction(race_id, noise=1.)
        race_name = race_tracker.race_details.DisplayName
        race_start = pd.to_datetime(
            race_tracker.race_details.DateString
        ).tz_convert(CURRENT_TIMEZONE)
        logger.info(f"loading {race_name}, starting at {race_start}")
        return cls(race_tracker, **kwargs)

    @classmethod
    def load_last_race(cls, **kwargs):
        last_race = get_last_race_started()
        return cls.from_race_id(last_race.name, **kwargs)

    def live_notebook_dashboard(self):
        from IPython.display import display, clear_output
        for live_data, intermediates in self.race_tracker.stream_livedata():
            clear_output(wait=True)
            if len(live_data):
                self.update(live_data, intermediates)
                display(self.fig)
            else:
                print('no race data received')
                break
        else:
            clear_output(wait=False)

    def live_ion_dashboard(self, block=True):
        import matplotlib.pyplot as plt
        plt.ion()
        plt.show()
        for live_data, intermediates in self.race_tracker.stream_livedata():
            if len(live_data):
                current_data = get_current_data(live_data)
                logger.info(current_data)
                self.update(live_data, intermediates)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            else:
                logger.info('no race data received')
                break

        plt.show(block=block)

    def _init(self, **kwargs):
        self._init_fig(**kwargs)
        self._init_tables()
        self._init_behind()
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

        self.fig = fig
        self.left_axes, self.middle_axes, self.right_axes = all_axes

        self.axes = np.c_[
            self.left_axes, self.middle_axes, self.right_axes
        ]
        self.b_behind_ax = self.left_axes[0]
        self.l_behind_ax = self.middle_axes[0]
        self.p_behind_ax = self.right_axes[0]
        self.p_behind_ax.sharey(self.left_axes[0])

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
        self._init_finish_table()

    def _init_finish_table(self):
        bbox = self.table_bbox
        n = len(self.race_tracker.lane_country)
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

    def _init_behind(self):
        self.b_behind, self.l_behind = self._init_bar_line(
            self.b_behind_ax, self.l_behind_ax
        )

    def _init_flags(self):
        self.b_distance_flags = self.race_tracker.plot_flags(
            self.race_tracker.country_lane * 0,
            zoom=0.015,
            zorder=10,
            ax=self.b_behind_ax
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

        self.b_behind_ax.set_ylabel('time behind leader (s)')
        self.b_pace_ax.set_ylabel('split (/500m)')
        self.b_stroke_rates_ax.set_ylabel('rate (s/m)')
        self.axes[-1, 0].set_xlabel('country')
        self.axes[-1, 1].set_xlabel('distance (m)')

        self.axes[-1, 0].set_xticklabels(
            self.axes[-1, 0].get_xticklabels(),
            rotation=-45,
        )
        self._set_finish_axes()
        self.fig.tight_layout()
        self.fig.subplots_adjust(**self._subplots_adjust)

    def _set_right_axes(self):
        for ax in self.right_axes:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        self.right_axes[-1].set_xticks(
            self.right_axes[-1].get_xticks()
        )
        self.right_axes[-1].set_xticklabels(
            self.right_axes[-1].get_xticklabels(),
            rotation=45,
        )

    def _set_pred_axes(self):
        logger.debug('setting prediction axes labels')
        self._set_right_axes()
        self.p_behind_ax.set_ylabel('predicted finish')
        self.p_pace_ax.set_ylabel('predicted time')
        self.p_win_ax.set_ylabel('probability of win')
        self._set_right_axes()

    def _set_finish_axes(self):
        logger.debug('setting finish axes labels')
        self._set_right_axes()
        self.p_behind_ax.set_ylabel('finish time difference (s)')
        self.p_pace_ax.set_ylabel('average split (/500m)')
        self.p_win_ax.set_ylabel('finish position')
        self._set_right_axes()

    def update(self, live_data=None, intermediates=None, predictions=None):
        live_data, intermediates = self.update_livedata(
            live_data, intermediates)
        if self.finished:
            logger.debug('creating finished axes')
            self.update_finish_axes(live_data, intermediates)
        else:
            logger.debug('updating prediction axes')
            self.update_predictions(
                predictions=predictions, live_data=live_data
            )

    def update_livedata(self, live_data=None, intermediates=None):
        if live_data is None:
            logger.debug("loading live data")
            live_data, intermediates = self.race_tracker.update_livedata()
        if intermediates is None:
            intermediates = self.race_tracker.intermediate_results

        if len(live_data):
            self.update_stroke_rate(live_data)
            self.update_pace(live_data)
            self.update_behind(live_data)
            self.update_pos(live_data)
            self.update_intermediates(intermediates)

        return live_data, intermediates

    def update_finish_axes(self, live_data, intermediates):
        axes = self.right_axes
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        self._set_finish_axes()
        self._init_finish_table()
        self.p_times = None
        self.p_finish = None
        self.p_win = None

        dist = 2000
        finish_times = intermediates[dist]
        final_pos = finish_times.sort_values()
        final_pos[:] = range(1, len(final_pos) + 1)

        # Plot finish postitions
        last = final_pos.max() + 1
        self.race_tracker.bar(
            final_pos - last, bottom=last,
            ax=axes[2]
        )
        axes[2].set_ylim(last, 0.5)
        axes[2].set_yticks(range(1, last))
        axes[2].set_xticklabels(
            axes[2].get_xticklabels(),
            rotation=45,
        )

        xlims = axes[2].get_xlim()
        # Plot finish delta lines
        finish_deltas = pd.DataFrame(
            [finish_times - finish_times.min()]*2,
            index=xlims
        )
        self.race_tracker.plot(finish_deltas, ax=axes[0])
        # Plot finish speeds
        average_pace = pd.DataFrame(
            [finish_times / dist * 500]*2,
            index=xlims
        )
        self.race_tracker.plot(average_pace, ax=axes[1])
        axes[1].set_ylim(*self.left_axes[1].get_ylim())
        format_yaxis_splits(axes[1])
        axes[1].set_ylim(*self.left_axes[1].get_ylim())
        axes[2].set_xlim(*xlims)

        # update right table
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

    def update_pos(self, live_data):
        logger.debug('update_pos')
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
        logger.debug('update_intermediates')
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
        else:
            self.finished = False

    def update_behind(self, live_data):
        logger.debug('update_behind')
        # self.race_tracker.update_bar(
        #     self.b_behind,
        #     live_data.timeFromLeader.iloc[-1],
        # )
        self.race_tracker.update_plot(
            self.l_behind,
            live_data.distanceTravelled,
            live_data.timeFromLeader,
        )
        self.race_tracker.update_flags(
            self.b_distance_flags,
            live_data.timeFromLeader.iloc[-1],
        )
        ymin = live_data.timeFromLeader.values.max() + 1
        self.b_behind_ax.set_ylim(
            ymin,
            -ymin*0.1,
        )

    def update_pace(self, live_data):
        logger.debug('update_pace')
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
        ylim = (pace.values[-100:-1].max() + 1, pace.values.min() - 1)
        self.b_pace_ax.set_ylim(ylim)
        format_yaxis_splits(self.b_pace_ax)
        self.b_pace_ax.set_ylim(ylim)

    def update_stroke_rate(self, live_data):
        logger.debug('update_stroke_rate')
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
        # self.plot_pred_distance(*preds_dist)
        self.plot_pred_pace(*preds_pace)
        self.plot_finish_times(*preds_time)
        self.plot_win_probs(win_probs)

    def update_predictions(self, predictions=None, live_data=None):
        logger.debug('update_predictions')
        if predictions is None:
            predictions = self.race_tracker.predict(live_data=live_data)

        if predictions:
            preds_pace, preds_time, preds_dist, win_probs = predictions
            self.update_pred_behind(*preds_time)
            self.update_pred_pace(*preds_pace)
            self.update_finish_times(*preds_time)
            self.update_win_probs(win_probs)

        self._set_pred_axes()

    def update_pred_behind(self, pred_time, pred_time_std):

        leader_time = pred_time.min(1)
        pred_behind = pred_time - leader_time.values[:, None]
        if self.p_finish is None:
            logger.debug('plot_pred_finish_behind')
            self.right_axes[0].clear()
            self._init_finish_table()

            self.p_finish = \
                self.race_tracker.plot_finish(
                    pred_behind.iloc[-1],
                    pred_time_std.iloc[-1],
                    set_lims=False,
                    ax=self.right_axes[0],
                )
        else:
            logger.debug('update_pred_finish_behind')
            self.race_tracker.update_plot_finish(
                *self.p_finish,
                pred_behind.iloc[-1],
                pred_time_std.iloc[-1],
            )

        if self.p_behind is None:
            logger.debug('plot_pred_behind')
            self.p_behind = \
                self.race_tracker.plot_uncertainty(
                    pred_behind, pred_time_std,
                    ax=self.middle_axes[0], alpha=self.alpha
                )
        else:
            logger.debug('update_pred_behind')
            self.race_tracker.update_plot_uncertainty(
                *self.p_behind,
                pred_behind, pred_time_std,
            )

        if self.race_tracker.completed:
            ymax = pred_behind.values.max()
        else:
            ymax = pred_behind.iloc[-1].max()

        self.right_axes[0].set_ylim(ymax, -ymax * 0.1)

    def update_pred_distance(self, pred_dist, pred_dist_std):
        logger.debug('update_pred_distance')
        pred_dist_behind = pred_dist.max(1).values[:, None] - pred_dist
        if self.p_behind is None:
            # self.right_axes[0].clear()
            self.p_behind = \
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
        else:
            pred_dist_behind = pred_dist.max(1).values[:, None] - pred_dist
            self.race_tracker.update_plot_uncertainty(
                *self.p_behind,
                pred_dist_behind, pred_dist_std,
            )
            self.race_tracker.update_plot_finish(
                *self.p_finish,
                pred_dist_behind.iloc[-1],
                pred_dist_std.iloc[-1],
            )

        if self.race_tracker.completed:
            ymax = pred_dist_behind.values.max() + 5
        else:
            ymax = pred_dist_behind.iloc[-1].max() + 5
        self.right_axes[0].set_ylim(ymax, -9)

    def update_pred_pace(self, pred_pace, pred_pace_std):
        logger.debug('update_pred_pace')
        if self.p_pace is None:
            self.p_pace = \
                self.race_tracker.plot_uncertainty(
                    pred_pace, pred_pace_std,
                    ax=self.middle_axes[1], alpha=self.alpha
                )
        else:
            self.race_tracker.update_plot_uncertainty(
                *self.p_pace, pred_pace, pred_pace_std,
            )

        ylim = (
            (pred_pace + pred_pace_std).values[-100:].max(),
            (pred_pace - pred_pace_std).values.min())
        self.b_pace_ax.set_ylim(ylim)
        format_yaxis_splits(self.b_pace_ax)
        self.b_pace_ax.set_ylim(ylim)

    def update_finish_times(self, pred_times, pred_times_std):
        logger.debug('update_finish_times')
        if self.finished:
            return None

        if self.p_times is None:
            self.right_axes[1].clear()
            ax = self.right_axes[1]
            self.p_times = \
                self.race_tracker.plot_finish(
                    pred_times.iloc[-1], pred_times_std.iloc[-1],
                    ax=ax
                )
            ylims = ax.get_ylim()
            format_yaxis_splits(ax)
            ax.set_ylim(*ylims[::-1])

        else:
            from matplotlib.ticker import AutoLocator
            ylims = self.race_tracker.update_plot_finish(
                *self.p_times, pred_times.iloc[-1], pred_times_std.iloc[-1],
            )
            self.right_axes[1].yaxis.set_major_locator(AutoLocator())
            format_yaxis_splits(self.right_axes[1])
            self.right_axes[1].set_ylim(*ylims[::-1])

        countries = self.race_tracker.lane_country
        finish_times = pred_times.iloc[-1]
        finish_pos = finish_times.sort_values()
        finish_pos[:] = np.arange(1, len(finish_pos) + 1)
        pgmt = (self.race_tracker.gmt / finish_times).apply("{:.1%}".format)
        update_table(
            self.right_table,
            countries,
            ['PGMT', 'pos'],
            pd.DataFrame({
                'PGMT': pgmt[countries],
                'pos': finish_pos[countries].astype(int),
            })
        )

    def update_win_probs(self, win_probs):
        logger.debug('update_win_probs')
        if self.p_win is None:
            self.right_axes[2].clear()
            self.p_win = self.race_tracker.bar(
                win_probs, ax=self.p_win_ax
            )
        else:
            self.race_tracker.update_bar(self.p_win, win_probs)

        if win_probs.notna().any():
            ymax = win_probs.dropna().max() + 0.05
        else:
            ymax = 0.05
        self.p_win_ax.set_ylim(0, ymax)


def main(block=True):
    import matplotlib.pyplot as plt
    from world_rowing.api import show_next_races

    dash = Dashboard.load_last_race(figsize=(12, 8))
    dash.live_ion_dashboard(block=False)

    logger.info('race results:')
    logger.info(dash.race_tracker.intermediate_results)

    logger.info('\nupcoming races:')
    logger.info(show_next_races())
    plt.show(block=block)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
