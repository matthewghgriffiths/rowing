

import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def plot_heart_rates(
    time_above_hr, hrs=None, hr_to_plot=None, hr_colors=None, 
    n_hr_labels=10, cmap='hot_r', side_text=True, ax=None, 
):
    if hrs is None:
        hrs = time_above_hr.columns.sort_values()
    if hr_to_plot is None:
        cumtime = time_above_hr.sum()[hrs[::-1]]
        hr_to_plot = cumtime.index[
            cumtime.searchsorted(np.linspace(0, cumtime.max(), n_hr_labels + 1))
        ][:0:-1]
    if hr_colors is None:
        hr_colors = dict(zip(
            hrs, plt.cm.get_cmap(cmap)(hrs)
        ))

    ax = ax or plt.gca()

    fill_betweens = {}
    base = pd.Series(0., index=time_above_hr.index)
    for hr in hrs[::-1]:
        time = time_above_hr[hr]
        fill_betweens[hr] = ax.fill_between(
            base.index, base, time, color=hr_colors[hr]
        )
        base = time 

    lines = {}
    for hr in hr_to_plot:
        lines[hr] = ax.plot(
            base.index, 
            time_above_hr[hr], 
            label=hr,
            color=hr_colors[hr],
            lw=2,
            path_effects=[
                pe.Stroke(linewidth=5, foreground='k'), 
                pe.Normal()
            ]
        )

    if side_text:
        for hr, time in time_above_hr.loc[
            time_above_hr.index[-1], hr_to_plot
        ].items():
            ax.text(
                time_above_hr.index[-1], time, f" {hr}"
            )

    return fill_betweens, lines, hr_to_plot