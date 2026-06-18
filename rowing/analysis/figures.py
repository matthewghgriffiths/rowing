"""Pure figure builders and export helpers for the rowing apps.

Plotly figure construction and zip/parquet export. No Streamlit calls and no
caching, so they can be imported and tested directly; ``rowing.analysis.app``
re-exports them (and wraps the cached one) for the Streamlit layer.
"""

import datetime
import io
import logging
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def outlier_range(data, quantiles=(0.05, 0.5, 0.9)):
    y0, y1, y2 = data.quantile(quantiles)
    r = (y2 - y0) * 0.1
    dt = max(y2 - y1, y1 - y0) * 1.1
    yrange = (max(y1 - dt, data.min() - r), min(y1 + dt, data.max() + r))
    return yrange


def scatter(data, x, y, fig=None, **kwargs):
    fig = fig or go.Figure()

    xdata = data[x]
    ydata = data[y]

    xaxis = "xaxis" + kwargs.get("xaxis", "x")[1:]
    yaxis = "yaxis" + kwargs.get("yaxis", "y")[1:]
    yaxis_layout = dict(title=dict(text=y))
    xaxis_layout = dict(title=dict(text=x))
    if yaxis != "yaxis":
        yaxis_layout.update(
            side="right",
            tickmode="sync",
            overlaying="y",
            autoshift=True,
            automargin=True,
        )

    if x_is_td := pd.api.types.is_timedelta64_dtype(xdata):
        xdata = xdata + pd.Timestamp(0)
        xaxis_layout.update(tickformat="%-M:%S", range=outlier_range(xdata))
    if y_is_td := pd.api.types.is_timedelta64_dtype(ydata):
        ydata = ydata + pd.Timestamp(0)
        yaxis_layout.update(tickformat="%-M:%S", range=outlier_range(ydata))
    if y_is_obj := pd.api.types.is_object_dtype(ydata):
        (t,) = ydata.map(type).mode()
        if t == datetime.time:
            t0 = pd.Timestamp(0)
            ydata = ydata.map(
                lambda t: (
                    pd.Timestamp(
                        year=t0.year,
                        month=t0.month,
                        day=t0.day,
                        hour=t.hour,
                        minute=t.minute,
                        second=t.second,
                        microsecond=t.microsecond,
                    )
                    if pd.notna(t)
                    else pd.Timestamp(np.nan)
                )
            )
            yaxis_layout.update(tickformat="%HH:%MM")

    kwargs.setdefault("name", y)
    if "text" not in kwargs:
        kwargs["text"] = (data.apply((("%s={0.%s} " % (x, x)) + ("%s={0.%s}" % (y, y))).format, axis=1),)

    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=ydata,
            **kwargs,
        )
    )
    fig.update_layout(
        {
            yaxis: yaxis_layout,
            xaxis: xaxis_layout,
        }
    )
    return fig


def piece_names(data, name="name", leg="leg"):
    pieces = data.groupby([name, leg]).size().rename("count").reset_index()[[name, leg]]
    pieces = pieces.join(pieces.groupby(name).size().rename("n_legs"), on=name)
    pieces["piece"] = pieces[name] + np.select(
        pieces.n_legs == 1, pieces[leg].apply("".format), pieces[leg].apply(" leg={}".format)
    )
    return pieces


def make_telemetry_distance_figure(compare_power, landmark_distances, col, facet_col_wrap=4):
    n_legs = compare_power.groupby(["name", "leg"]).size().groupby(level=0).size()

    if col == "Work PC":
        WorkPC_cols = ["Work PC Q1", "Work PC Q2", "Work PC Q3", "Work PC Q4"]
        pc_work = compare_power[["name", "leg", "Distance", "Position"] + WorkPC_cols].copy()
        pc_work["piece"] = pc_work.name + np.select(
            n_legs.loc[pc_work.name] == 1, pc_work.leg.apply("".format), pc_work.leg.apply(" leg={}".format)
        )
        pc_work["R"] = pc_work["piece"].str.cat(pc_work.Position, sep="|")
        pc_plot_work = pc_work.set_index(["Distance", "R"])[WorkPC_cols].stack().rename(col).reset_index()

        fig = px.area(
            pc_plot_work,
            x="Distance",
            y=col,
            facet_col="R",
            facet_col_wrap=facet_col_wrap,
            color="Measurement",
            facet_col_spacing=0.01,
            facet_row_spacing=0.02,
            template="plotly_white",
        )
    else:
        fig = go.Figure()
        for (file, leg), data in compare_power.groupby(["name", "leg"]):
            if col in data:
                cols = data[[col]].columns
                pos_power = data.dropna(subset=cols, how="all").groupby(["Position", "Side"])

                for (pos, side), pos_data in pos_power:
                    name = f"{file} {side}" if n_legs[file] == 1 else f"{file} {side} {leg=:d}"
                    for c in cols:
                        fig.add_trace(
                            go.Scatter(
                                x=pos_data["Distance"],
                                y=pos_data[c],
                                legendgroup=f"{file} {leg} {side}",
                                legendgrouptitle_text=name,
                                name=f"{pos}",
                                mode="lines",
                            )
                        )
            else:
                logger.warning("%s not in data; columns=%s", col, list(data.columns))

        fig.update_layout(
            xaxis_title="Distance (km)",
            yaxis_title=col,
        )

    for landmark, distance in landmark_distances.items():
        fig.add_vline(x=distance, annotation_text=landmark, annotation=dict(textangle=-90))
    return fig


def figures_to_zipfile(figures, file_type, **kwargs):
    zipdata = io.BytesIO()
    with zipfile.ZipFile(zipdata, "w") as zipf:
        for name, fig in figures.items():
            if file_type == "html":
                fig_data = fig.to_html(**kwargs)
            else:
                fig_data = fig.to_image(format=file_type, **kwargs)

            zipf.writestr(f"{name}.{file_type}", fig_data)

    zipdata.seek(0)
    return zipdata


def telemetry_to_zipfile(telemetry_data):
    zipdata = io.BytesIO()
    with zipfile.ZipFile(zipdata, "w") as zipf:
        for name, piece_data in telemetry_data.items():
            for k, data in piece_data.items():
                if isinstance(data, pd.DataFrame):
                    save_data = data.copy()
                elif isinstance(data, pd.Series):
                    save_data = data.reset_index()

                for c, vals in save_data.items():
                    if pd.api.types.is_object_dtype(vals.dtype):
                        save_data[c] = vals.astype(str)

                with zipf.open(f"{name}/{k}.parquet", "w") as f:
                    save_data.to_parquet(f, index=False)

    zipdata.seek(0)
    return zipdata
