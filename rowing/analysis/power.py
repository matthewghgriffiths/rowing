
import numpy as np
import pandas as pd

from scipy import signal, integrate

intervals = np.r_[
    np.arange(0, 10, 1),
    np.arange(10, 30, 5),
    np.arange(30, 60, 10),
    np.arange(60, 300, 30),
    np.arange(300, 1800, 60),
    np.arange(1800, 3600, 120),
    np.arange(3600, 7200, 300),
]


def max_mean_power(P, t, intervals):
    t = np.array(t)
    W = integrate.cumulative_trapezoid(
        P, t, axis=0, initial=0)
    if W.ndim == 1:
        W = W[:, None]

    j, i = np.triu_indices(t.size, k=1)
    dt = t[i] - t[j]
    dW = W[i] - W[j]

    Pdiffs = pd.DataFrame(
        dW / dt[:, None],
        index=dt,
        columns=P.columns if isinstance(P, pd.DataFrame) else None
    )
    intervals = np.array(intervals)
    Pbest = Pdiffs.groupby(
        pd.cut(Pdiffs.index, intervals),
        observed=True,
    ).max().dropna(how='all').sort_index(ascending=False).cummax()
    Pbest.index = Pbest.index.map(
        lambda x: x.left).astype(intervals.dtype)
    return Pbest


def identify_pieces(
        stroke_power,
        n_stroke_avg: int = 10,
        min_rating: float = 10,
        min_duration: float = 30,
        min_power: float = 50,
        start_height=2,
        end_height=5,
        peak_distance=10,
):

    stroke_times = stroke_power.index.to_series().fillna(0) / 1000
    durations = - stroke_times.diff(-1).ffill()
    rating = (60 / durations).where(
        stroke_power.notna() | (stroke_power < min_power), 0)
    stopped = rating < min_rating

    # Detect start and end of rate changes
    smooth_rating = (
        rating.rolling(n_stroke_avg, min_periods=1).mean().fillna(0)
    ).where(~stopped, 0)
    start_detect = smooth_rating.diff(n_stroke_avg).fillna(0)
    end_detect = smooth_rating.diff(-n_stroke_avg).fillna(0)
    startp, info = signal.find_peaks(
        start_detect, height=start_height, distance=peak_distance)
    endp, info = signal.find_peaks(
        end_detect, height=end_height, distance=peak_distance)

    starts = pd.Series(0, rating.index)
    starts.iloc[startp] = 1
    # starts.loc[stopped.shift(fill_value=False)] = 1

    starti = starts.index.get_indexer_for(starts.index[starts == 1])

    stops = pd.Series(0, rating.index)
    stops.iloc[endp] = 1
    stops.loc[stopped] = 1
    stopi = stops.index.get_indexer_for(stops.index[stops == 1])

    piece = starts.rename('piece') * np.nan
    piece.iloc[0] = 0
    piece.iloc[starti] = 1
    piece.iloc[stopi] = 0

    piece = piece.ffill()
    piece.iloc[starti] = 1 + (piece.iloc[starti - 1] == 0).cumsum()
    piece = piece.where(piece != 1).ffill().astype(int)

    n_piece = 0
    while (n := piece.nunique()) != n_piece:
        n_piece = n
        piece = piece.where(
            (rating * durations).groupby(piece).transform('sum')
            / durations.groupby(piece).transform('sum') > min_rating, 0)
        piece = piece.where(
            durations.groupby(piece).transform('sum') > min_duration, 0)
        piece = piece.where(
            (stroke_power * durations).groupby(piece).transform('sum')
            / durations.groupby(piece).transform('sum') > min_power, 0)

    return piece


def piece_averages(data, **kwargs):
    power = data.SwivelPower.mean(axis=1)
    piece = identify_pieces(power, **kwargs)
    stroke_times = data.index.to_series().fillna(0) / 1000
    durations = - stroke_times.diff(-1).ffill()

    piece_avg = (
        (data.T * durations).T.groupby(piece).sum().T
        / durations.groupby(piece).sum()
    ).T

    piece_avg[('Timestamp', 'Boat')] = pd.to_timedelta(
        data.index.to_series().groupby(piece).min(), unit='ms')
    piece_avg[('Start', 'm')] = data.Distance.Boat.groupby(
        piece).min().round(2)
    piece_avg[('Strokes', 'count')] = data.Distance.Boat.groupby(piece).size()
    piece_avg[('Length', 'm')] = data.Distance.Boat.groupby(
        piece).apply(np.ptp)
    piece_avg[('Duration', 's')] = durations.groupby(piece).sum()
    piece_avg[('Max Rating', '/min')] = data.Rating.Boat.groupby(piece).max()
    piece_avg[('Min Rating', '/min')] = data.Rating.Boat.groupby(piece).min()
    piece_avg[('AvgBoatSpeed', 'm/s')] = piece_avg[('Length', 'm')
                                                   ] / piece_avg[('Duration', 's')]
    boat_col_order = [
        'Timestamp', 'Length', 'Duration', 'Rating', 'Average Power', 'AvgBoatSpeed',
        'Min Rating', 'Max Rating',
        'Dist/Stroke', 'StrokeNumber', 'Start', 'Strokes'
    ]
    rower_col_order = [
        'SwivelPower', 'MinAngle', 'MaxAngle', 'CatchSlip', 'FinishSlip',
        'Drive Start T'
    ]
    return pd.concat([
        piece_avg[boat_col_order].round(2),
        piece_avg[rower_col_order].round(1),
    ], axis=1).drop(index=0)
