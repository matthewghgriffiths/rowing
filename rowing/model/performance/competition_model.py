from collections.abc import Callable
from typing import NamedTuple

import flax.nnx as nnx
import haiku as hk
import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp

# from scipy import sparse
from rowing.model.gp import kernels
from rowing.model.gp import utils as gp_utils
from rowing.model.gp.kernels import Hyper
from rowing.model.gp.utils import GPSystem
from rowing.world_rowing import fields


def year_to_date(year):
    y0 = np.floor(year).astype(int)
    start_date = pd.to_datetime(dict(year=y0, month=1, day=1))
    end_date = pd.to_datetime(dict(year=y0 + 1, month=1, day=1))
    date = start_date + (end_date - start_date) * (year % 1)
    return date


def get_athlete_kernel():
    return kernels.SumKernel(
        kernels.SEKernel(name="athlete_se_0"),
        kernels.Matern32(name="athlete_matern32", scale=1),
        kernels.Bias(name="athlete_bias"),
    )


def get_race_kernel():
    return kernels.SumKernel(
        kernels.Matern12(name="race_matern12_0"),
        kernels.SEKernel(name="race_kernel0"),
        # kernels.SEKernel(name='race_kernel1'),
        # kernels.Bias(name='race_bias'),
    )


def get_weather_kernel(n_features):
    return kernels.SumKernel(
        kernels.SEKernel(name="weather_se_0", shape=(n_features,)),
        kernels.DotProduct(name="weather_dot_0"),
        kernels.DotProduct(name="weather_dot_1") ** 2,
        # kernels.DotProduct(name='weather_dot_2')**3,
        # kernels.Matern32(name='weather_matern32_0', shape=(n_features,)),
        # kernels.Matern52(name='weather_matern52_0', shape=(n_features,)),
    )


def get_lane_kernel():
    return kernels.SumKernel(
        kernels.SEKernel(name="lane_kernel0", scale=2),
        # kernels.Matern12(name='lane_matern12_0', scale=3),
        # kernels.Bias(name='race_bias'),
    )


GetKernel = Callable[[], kernels.AbstractKernel]
GetKernelD = Callable[[int], kernels.AbstractKernel]


class AthleteModel(NamedTuple):
    years: jax.Array
    year0: jax.Array
    W_athlete: jax.Array
    gram_athlete: jax.Array
    athlete_kernel: GetKernel = get_athlete_kernel
    metadata: dict | None = None

    def get_kernels(self):
        years = self.years - self.year0
        K_athlete = self.athlete_kernel().K(years, years) * self.gram_athlete
        return (K_athlete,)


def boatclass_kernel(K):
    boatclass_var = jnp.exp(hk.get_parameter(fields.BoatType, [], init=jnp.zeros))
    return boatclass_var * K


class RaceModel(NamedTuple):
    hours: jax.Array
    W_venue: jax.Array
    W_boatclass: jax.Array
    W_lane: jax.Array
    gram_venue: jax.Array
    gram_boatclass: jax.Array
    gram_lane: jax.Array
    race_kernel: GetKernel = get_race_kernel
    lane_kernel: GetKernel | None = None
    metadata: dict | None = None

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel

    def get_kernels(self):
        times = self.hours
        K_race_times = self.race_kernel().K(times, times) * self.gram_venue
        if self.lane_kernel:
            gram_lane = jnp.where(jnp.isfinite(self.gram_lane), self.gram_lane, 0)
            K_lane = jnp.where(
                jnp.isfinite(self.gram_lane), self.lane_kernel().K(times, times) * self.gram_venue * gram_lane, 0
            )
            K_race_times += K_lane

        K_boatclass = boatclass_kernel(self.gram_boatclass)

        return K_race_times, K_boatclass


class RaceWeatherModel(NamedTuple):
    hours: jax.Array
    weather: jax.Array
    W_venue: jax.Array
    W_boatclass: jax.Array
    gram_venue: jax.Array
    gram_boatclass: jax.Array

    y: jax.Array | None = None
    race_kernel: GetKernel = get_race_kernel
    weather_kernel: GetKernelD = get_weather_kernel

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    def get_kernels(self):
        K_boatclass = boatclass_kernel(self.gram_boatclass)

        kernels = (K_boatclass,)

        times = self.hours
        if self.race_kernel:
            K_race_times = self.race_kernel().K(times, times) * self.gram_venue
            kernels += (K_race_times,)

        if self.weather_kernel:
            weather = self.weather
            K_weather = self.weather_kernel(weather.shape[1]).K(weather, weather)
            kernels += (K_weather,)

        return kernels

    @classmethod
    def from_conditions(cls, race_conditions, weather_cols, **kwargs):
        start_times = race_conditions.Race_Start_utc
        order = race_conditions.race_id

        weather = jnp.array(race_conditions[weather_cols].values)

        first_time = start_times.min()
        times = (start_times - first_time).dt.total_seconds().values
        hours = times / 60 / 60

        weights = {
            f: race_conditions.groupby(["race_id", f]).size().unstack(level=1, fill_value=0).loc[order]
            for f in [
                fields.Day,
                "race_event_competition_venueId",
                fields.race_boatClass,
                fields.BoatType,
            ]
        }
        Ws = {k: jnp.array(df.values) for k, df in weights.items()}
        grams = {k: W @ W.T for k, W in Ws.items()}

        return cls(
            hours=hours,
            weather=weather,
            gram_venue=grams["race_event_competition_venueId"],
            gram_boatclass=grams["Boat Class"],
            y=jnp.array(race_conditions.PGMT.values),
            **kwargs,
        )


class PerformanceModel(NamedTuple):
    athlete_model: AthleteModel
    race_model: RaceModel
    y: jax.Array
    metadata: dict | None = None

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    def get_kernels(self):
        return self.athlete_model.get_kernels() + self.race_model.get_kernels()

    @classmethod
    def from_data(
        cls,
        results,
        seats,
        athletes,
        athlete_kernel=get_athlete_kernel,
        race_kernel=get_race_kernel,
        lane_kernel=None,
        **kwargs,
    ):
        seats = seats.join(
            1 / seats.groupby("athletes_raceBoatId").size().rename("seat_weight"), on="athletes_raceBoatId"
        ).join(results["Boat Type"], on="athletes_raceBoatId")
        boat_order = results.index
        athlete_order = athletes.index

        weights = {
            f: results.groupby(["raceBoats_id", f]).size().unstack(level=1, fill_value=0).loc[boat_order]
            for f in [
                fields.Day,
                "race_event_competition_venueId",
                fields.race_boatClass,
                fields.BoatType,
            ]
        }
        weights["athlete"] = seats.seat_weight.unstack(level=1, fill_value=0).loc[boat_order, athlete_order]
        weights["lane"] = (
            (results.Lane - results.groupby("race_id").Lane.mean().loc[results.race_id].values).loc[boat_order].to_frame()
        )

        Ws = {k: jnp.array(df.values) for k, df in weights.items()}
        grams = {k: W @ W.T for k, W in Ws.items()}

        start_times = results[fields.race_Date]
        first_time = start_times.min()
        last_time = start_times.max()

        first_year = first_time.year + first_time.day_of_year / 365.25
        last_year = last_time.year + last_time.day_of_year / 365.25

        times = (start_times - first_time).dt.total_seconds().values
        hours = times / 60 / 60
        years = first_year + (last_year - first_year) * (times - times.min()) / (times.max() - times.min())

        year0 = first_year - 2

        race_model = RaceModel(
            hours=hours,
            W_boatclass=Ws["Boat Class"],
            W_venue=Ws["race_event_competition_venueId"],
            W_lane=Ws["lane"],
            gram_venue=grams["race_event_competition_venueId"],
            gram_boatclass=grams["Boat Class"],
            gram_lane=grams["lane"],
            race_kernel=race_kernel,
            lane_kernel=lane_kernel,
        )
        athlete_model = AthleteModel(
            years=years,
            year0=year0,
            W_athlete=Ws["athlete"],
            gram_athlete=grams["athlete"],
            athlete_kernel=athlete_kernel,
        )

        return cls(
            athlete_model=athlete_model,
            race_model=race_model,
            y=results.PGMT.values,
            metadata={
                "weights": weights,
                "results": results,
            },
        )


class PerformanceGP(nnx.Module, pytree=False):
    """Eager flax.nnx view of a :class:`PerformanceModel`.

    Instantiates the model's kernels once (so their params are persistent and shared
    across evaluations) and holds the boat-class and observation-noise scalars as
    ``nnx.Param``s. Exposes the same kernels / GP system / loss as the old Haiku flow;
    optimise by ``nnx.split(gp, nnx.Param)`` and updating the param state.
    """

    @classmethod
    def from_inputs(
        cls,
        mi,
        *,
        athlete_kernel=get_athlete_kernel,
        race_kernel=get_race_kernel,
        lane_kernel=None,
        params=None,
    ):
        """Build a PerformanceGP from a :class:`rowing.model.data.ModelInputs` atomic pytree.

        Re-expresses :meth:`PerformanceModel.from_data` over the atomic arrays: categorical Grams
        are equality-outer-products of integer codes, ``W_athlete`` is a seat-weight scatter, and
        ``year``/``hour`` are carried through. Optionally loads a legacy Haiku ``params`` dict.
        """
        athlete_model = AthleteModel(
            years=mi.year,
            year0=mi.year0,
            W_athlete=mi.W_athlete(),
            gram_athlete=mi.gram_athlete(),
            athlete_kernel=athlete_kernel,
        )
        race_model = RaceModel(
            hours=mi.hour,
            W_venue=mi.one_hot("venue"),
            W_boatclass=mi.one_hot("class"),
            W_lane=mi.W_lane,
            gram_venue=mi.categorical_gram("venue"),
            gram_boatclass=mi.categorical_gram("class"),
            gram_lane=mi.gram_lane(),
            race_kernel=race_kernel,
            lane_kernel=lane_kernel,
        )
        model = PerformanceModel(athlete_model=athlete_model, race_model=race_model, y=mi.y)
        gp = cls(model)
        if params is not None:
            load_haiku_params(gp, params)
        return gp

    def __init__(self, model: "PerformanceModel"):
        am, rm = model.athlete_model, model.race_model
        self.athlete_kernel = am.athlete_kernel()
        self.race_kernel = rm.race_kernel()
        self.lane_kernel = rm.lane_kernel() if rm.lane_kernel is not None else None
        self.boatclass_var = Hyper(None)
        self.log_noise = nnx.Param(jnp.zeros((), jnp.float64))

        # data (static, not optimised)
        self.years = am.years
        self.year0 = am.year0
        self.gram_athlete = am.gram_athlete
        self.W_athlete = am.W_athlete
        self.hours = rm.hours
        self.gram_venue = rm.gram_venue
        self.gram_lane = rm.gram_lane
        self.gram_boatclass = rm.gram_boatclass
        self.y = model.y
        # per-result observation weight (1 = homoscedastic); fit_robust shrinks it for outliers
        self.obs_weight = jnp.ones(jnp.shape(self.y))

    def get_kernels(self):
        years = self.years - self.year0
        K_athlete = self.athlete_kernel.K(years, years) * self.gram_athlete

        K_race = self.race_kernel.K(self.hours, self.hours) * self.gram_venue
        if self.lane_kernel is not None:
            gram_lane = jnp.where(jnp.isfinite(self.gram_lane), self.gram_lane, 0)
            K_race = K_race + jnp.where(
                jnp.isfinite(self.gram_lane),
                self.lane_kernel.K(self.hours, self.hours) * self.gram_venue * gram_lane,
                0,
            )

        K_boatclass = self.boatclass_var.value * self.gram_boatclass
        return K_athlete, K_race, K_boatclass

    def get_full_kernel(self):
        return sum(self.get_kernels())

    def get_jitter_kernel(self):
        K = self.get_full_kernel()
        # heteroscedastic observation noise: sigma^2 / obs_weight per result (down-weights outliers)
        return K + jnp.diag(jnp.exp(self.log_noise[...]) / self.obs_weight)

    def fit_robust(self, nu=4.0, n_iter=8):
        """Estimate per-result observation weights under a Student-t likelihood (IRLS), in place.

        Iterates: fit the GP with the current per-result noise sigma^2 / w, form the studentised
        leave-one-out residual z_i^2 = a_i^2 / (K^-1)_ii, and set w_i = (nu+1)/(nu + z_i^2). Large
        residuals (bad/anomalous races) get small w_i -> inflated noise -> down-weighted. Returns w.
        """
        import numpy as np
        from scipy import linalg as sla

        y = np.asarray(self.y)
        K = np.asarray(self.get_full_kernel())
        base = float(np.exp(np.asarray(self.log_noise[...])))
        n = len(y)
        w = np.ones(n)
        eye = np.eye(n)
        for _ in range(n_iter):
            L = np.linalg.cholesky(K + np.diag(base / w))
            a = sla.cho_solve((L, True), y)
            # diag(K^-1) = column sum of (L^-1)^2 -- one triangular solve, no full inverse
            Linv = sla.solve_triangular(L, eye, lower=True)
            iKii = np.einsum("ij,ij->j", Linv, Linv)
            z2 = a**2 / iKii  # studentised leave-one-out residual, squared
            w = (nu + 1.0) / (nu + z2)
        self.obs_weight = jnp.asarray(w)
        return w

    def fit_loo(self, **min_kws):
        """Fit hyperparameters by maximising the leave-one-out predictive density, in place.

        A predictive (generalisation) objective -- preferred over the marginal likelihood, which can
        overfit (shortening the athlete length-scale and zeroing the bias, which hurt held-out
        ranking). Returns the scipy optimise result.
        """
        from rowing.model.gp.utils import fit_module

        return fit_module(self, loss_fn=lambda m: -m.gp_system().loo_log_density(), **min_kws)

    def gp_system(self):
        return GPSystem.from_gram(self.get_jitter_kernel(), self.y)

    def loss(self):
        return self.gp_system().loss()

    def predict_athletes_scores(self, times, athletes_index=None, system=None):
        """Posterior mean athlete score at each ``time`` (years), one column per time."""
        if system is None:
            system = self.gp_system()
        K_pred = self.athlete_kernel.K(times, self.years)
        return pd.DataFrame(
            jnp.einsum("ij,jk,j->ki", K_pred, self.W_athlete, system.a),
            index=athletes_index,
            columns=times,
        )

    def predict_athletes_score(self, start, system=None):
        """Posterior mean and covariance of athlete scores at a single time ``start``."""
        if system is None:
            system = self.gp_system()
        k_pred = self.athlete_kernel.K(np.r_[start], self.years)[0]
        k00 = self.athlete_kernel.K(np.r_[start], np.r_[start])

        y_ath = jnp.einsum("j,jk,j->k", k_pred, self.W_athlete, system.a)
        Cov_ath = k00 * np.eye(self.W_athlete.shape[1]) - jnp.einsum(
            "j,ji,jk,kl,k->il",
            k_pred,
            self.W_athlete,
            system.inv_K(),
            self.W_athlete,
            k_pred,
        )
        return y_ath, Cov_ath


def predict_boat_scores(y_ath, cov_ath, athletes, athlete_ids, noise=1e-4):
    boat_ids = pd.Index(athletes.boatId.unique()).sort_values()

    boat_athlete_W = np.zeros((boat_ids.size, athlete_ids.size))
    boat_athlete_W[
        boat_ids.get_indexer_for(athletes[athletes.athletePosition != "c"].boatId),
        athlete_ids.get_indexer_for(athletes[athletes.athletePosition != "c"].personId),
    ] = 1
    w1 = boat_athlete_W.sum(1, keepdims=True)
    boat_athlete_W /= np.where(w1 > 0, w1, 1)

    y_boat = pd.Series(boat_athlete_W @ y_ath, index=boat_ids)
    cov_boat = pd.DataFrame(
        boat_athlete_W @ cov_ath @ boat_athlete_W.T + np.eye(len(boat_ids)) * noise,
        index=boat_ids,
        columns=boat_ids,
    )
    return y_boat, cov_boat


# Legacy Haiku param key -> nnx Hyper attribute on the kernel.
_HAIKU_KEY_TO_HYPER = {
    "log_var": "variance",
    "log_scale": "scale",
    "log_period": "period",
    "offset": "offset",
    "bias": "bias",
    "t0": "t0",
}


def _iter_named_kernels(kernel):
    """Yield every kernel in a (possibly composite) kernel tree that carries a ``name``."""
    if getattr(kernel, "name", None) is not None:
        yield kernel
    for sub in getattr(kernel, "kernels", []):  # Sum/Product kernels
        yield from _iter_named_kernels(sub)
    inner = getattr(kernel, "kernel", None)  # Slice/Power kernels
    if inner is not None:
        yield from _iter_named_kernels(inner)


def load_kernel_haiku_params(kernel, params: dict):
    """Copy legacy Haiku params onto a single (possibly composite) nnx kernel, in place.

    ``params`` is the flat ``{kernel_name: {log_var, log_scale, ...}}`` dict; each named sub-kernel
    of ``kernel`` receives the matching stored (log-space) values. Fixed Hypers (e.g. a pinned
    ``scale``) and unmatched keys are skipped. Returns ``kernel``.
    """
    for k in _iter_named_kernels(kernel):
        entry = params.get(k.name, {})
        for hk_key, attr in _HAIKU_KEY_TO_HYPER.items():
            if hk_key in entry:
                hyper = getattr(k, attr, None)
                if hyper is not None and hyper.param is not None:
                    hyper.param = nnx.Param(jnp.asarray(entry[hk_key], dtype=jnp.float64))
    return kernel


def load_haiku_params(gp: "PerformanceGP", params: dict) -> "PerformanceGP":
    """Load a legacy Haiku ``params.yaml`` dict into an nnx :class:`PerformanceGP`, in place.

    The old format is a flat dict keyed by kernel ``name`` (``{name: {log_var, log_scale, ...}}``)
    plus a ``'~'`` root holding ``Boat Type`` (boat-class log-variance) and ``log_noise``. The
    kernels were created with matching ``name=`` arguments, so we walk the model's kernels and
    copy each stored (log-space) value onto the corresponding learnable Hyper. Hypers that were
    fixed at construction (e.g. a pinned ``scale``) are skipped, as are param keys with no match.
    """

    def set_param(hyper, value):
        if hyper is not None and hyper.param is not None:
            hyper.param = nnx.Param(jnp.asarray(value, dtype=jnp.float64))

    for kernel in (gp.athlete_kernel, gp.race_kernel, gp.lane_kernel):
        if kernel is not None:
            load_kernel_haiku_params(kernel, params)

    root = params.get("~", {})
    if fields.BoatType in root:
        set_param(gp.boatclass_var, root[fields.BoatType])
    if "log_noise" in root:
        gp.log_noise = nnx.Param(jnp.asarray(root["log_noise"], dtype=jnp.float64))

    return gp


def dump_haiku_params(gp: "PerformanceGP") -> dict:
    """Extract a PerformanceGP's learnable hyperparameters into the legacy ``params.yaml`` dict.

    Inverse of :func:`load_haiku_params`: walks the named kernels and reads each learnable Hyper's
    (log-space) param, plus the ``'~'`` root (boat-class log-variance, ``log_noise``). Round-trips
    with the loader, so an optimised model can be saved and reused per competition.
    """
    hyper_to_key = {attr: key for key, attr in _HAIKU_KEY_TO_HYPER.items()}
    out: dict = {}
    for kernel in (gp.athlete_kernel, gp.race_kernel, gp.lane_kernel):
        if kernel is None:
            continue
        for k in _iter_named_kernels(kernel):
            entry = {}
            for attr, key in hyper_to_key.items():
                hyper = getattr(k, attr, None)
                if hyper is not None and hyper.param is not None:
                    entry[key] = float(np.asarray(hyper.param[...]))
            if entry:
                out[k.name] = entry
    out["~"] = {
        fields.BoatType: float(np.asarray(gp.boatclass_var.param[...])),
        "log_noise": float(np.asarray(gp.log_noise[...])),
    }
    return out


class CompetitionModel(NamedTuple):
    hours: np.ndarray
    years: np.ndarray
    year0: np.ndarray
    W_venue: np.ndarray
    W_athlete: np.ndarray
    W_boatclass: np.ndarray
    W_lane: np.ndarray
    y: np.ndarray
    gram_venue: np.ndarray
    gram_athlete: np.ndarray
    gram_boatclass: np.ndarray
    gram_lane: np.ndarray
    athlete_kernel: GetKernel = get_athlete_kernel
    race_kernel: GetKernel = get_race_kernel
    lane_kernel: GetKernel | None = get_race_kernel
    metadata: dict | None = None

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    @classmethod
    def from_data(cls, results, seats, athletes, **kwargs):
        seats = seats.join(1 / seats.groupby(level=0).size().rename("seat_weight"), on="athletes_raceBoatId").join(
            results["Boat Type"], on="athletes_raceBoatId"
        )
        boat_order = results.index
        athlete_order = athletes.index

        weights = {
            f: results.groupby(["raceBoats_id", f]).size().unstack(level=1, fill_value=0).loc[boat_order]
            for f in [
                fields.Day,
                "race_event_competition_venueId",
                fields.race_boatClass,
                fields.BoatType,
            ]
        }
        weights["athlete"] = seats.seat_weight.unstack(level=1, fill_value=0).loc[boat_order, athlete_order]
        # Make lanes 0 mean per race.
        weights["lane"] = (
            (results.Lane - results.groupby("race_id").Lane.mean().loc[results.race_id].values).loc[boat_order].to_frame()
        )

        Ws = {k: jnp.array(df.values) for k, df in weights.items()}
        grams = {k: W @ W.T for k, W in Ws.items()}

        start_times = results[fields.race_Date]
        first_time = start_times.min()
        last_time = start_times.max()

        first_year = first_time.year + first_time.day_of_year / 365.25
        last_year = last_time.year + last_time.day_of_year / 365.25

        times = (start_times - first_time).dt.total_seconds().values
        hours = times / 60 / 60
        years = first_year + (last_year - first_year) * (times - times.min()) / (times.max() - times.min())

        year0 = first_year - 2
        return cls(
            hours=hours,
            years=years,
            year0=year0,
            W_venue=Ws["race_event_competition_venueId"],
            W_athlete=Ws["athlete"],
            W_boatclass=Ws["Boat Class"],
            W_lane=Ws["lane"],
            y=results.PGMT.values,
            gram_venue=grams["race_event_competition_venueId"],
            gram_athlete=grams["athlete"],
            gram_boatclass=grams["Boat Class"],
            gram_lane=grams["lane"],
            metadata={
                "weights": weights,
            },
            **kwargs,
        )

    def get_kernels(self):
        boatclass_var = jnp.exp(hk.get_parameter(fields.BoatType, [], init=jnp.zeros))

        times = self.hours
        years = self.years - self.year0

        K_race_times = self.race_kernel().K(times, times) * self.gram_venue
        if self.lane_kernel:
            K_race_times += self.lane_kernel().K(times, times) * self.gram_venue * self.gram_lane
        K_athlete_times = self.athlete_kernel().K(years, years) * self.gram_athlete
        K_boatclass = boatclass_var * self.gram_boatclass

        return (
            K_race_times,
            K_athlete_times,
            K_boatclass,
        )

    get_full_kernel = gp_utils.get_full_kernel
    get_jitter_kernel = gp_utils.get_jitter_kernel
    gp_system = gp_utils.gp_system
    loss = gp_utils.loss

    # def get_full_kernel(self):
    #     K_race_times, K_athlete_times, K_boatclass = self.get_kernels()
    #     return (
    #         K_race_times
    #         + K_athlete_times
    #         + K_boatclass
    #     )

    # def get_jitter_kernel(self):
    #     K = self.get_full_kernel()
    #     race_var = jnp.exp(hk.get_parameter(
    #         "race_logvar", [], init=jnp.zeros, dtype=jnp.float64))
    #     K_noise = np.eye(len(K)) * race_var
    #     return K + K_noise

    # def gp_system(self):
    #     K = self.get_jitter_kernel()
    #     y = self.y
    #     return gp_utils.GPSystem.from_gram(K, y)

    # def loss(self):
    #     return self.gp_system().loss()


def filter_results(
    senior_data,
    years=None,
    keep_phases=None,
    min_races=None,
    min_racesize=None,
    min_pgmt=None,
    max_pgmt=1,
    keep_athletes=None,
    **kwargs,
):
    results = senior_data["results"]
    athletes = senior_data["athletes"]
    seats = senior_data["seats"]

    sel_results = results
    filtered = pd.Series(False, athletes.athletes_personId)
    filtered_raceBoats = set()
    for i in range(20):
        sel = (
            ~results.raceBoats_id.isin(filtered_raceBoats)
            & np.isfinite(results.PGMT)
            & results["Race Start"].notna()
            & results.raceBoats_id.isin(seats.athletes_raceBoatId)
        )
        if years:
            sel &= results.year.isin(years)
        if keep_phases:
            sel &= results.Phase.isin(keep_phases)
        if min_pgmt:
            sel &= results.PGMT > min_pgmt
        if max_pgmt:
            sel &= results.PGMT <= max_pgmt
        if min_racesize:
            sel &= (sel_results.groupby("race_id").size() > min_racesize).reindex(results.race_id, fill_value=False).values
        for k, filter in kwargs.items():
            sel &= filter(results[k])

        sel_results = results[sel].set_index("raceBoats_id").sort_values("Race Start")
        sel_seats = (
            seats[seats.athletes_raceBoatId.isin(sel_results.index) & (seats.athletes_boatPosition != "c")]
            .set_index(
                [
                    "athletes_raceBoatId",
                    "athletes_personId",
                ]
            )
            .sort_index()
        )

        update = sel_seats.groupby(level=1).size() < min_races
        filtered.update(update)
        if keep_athletes is not None:
            filtered[keep_athletes] = False
        if not update.any():
            break

        filtered_raceBoats = seats.athletes_raceBoatId[seats.athletes_personId.isin(filtered.index[filtered])]

    sel_athletes = (
        athletes[athletes.athletes_personId.isin(sel_seats.index.levels[1])].set_index("athletes_personId").sort_index()
    )

    return {"athletes": sel_athletes, "results": sel_results, "seats": sel_seats, "competitions": senior_data["competitions"]}


# class CompetitionModel(NamedTuple):
#     hours0: np.ndarray
#     hours1: np.ndarray
#     years0: np.ndarray
#     years1: np.ndarray
#     y: np.ndarray
#     gram_venue: np.ndarray
#     gram_athlete: np.ndarray
#     gram_boatclass: np.ndarray
#     athlete_kernel: Callable[[], kernels.AbstractKernel] = get_athlete_kernel
#     race_kernel: Callable[[], kernels.AbstractKernel] = get_race_kernel
#     metadata: Optional[Dict] = None

#     get_full_kernel = get_full_kernel
#     get_jitter_kernel = get_jitter_kernel
#     gp_system = gp_system
#     loss = loss

#     @classmethod
#     def from_data(cls, results, seats, athletes, **kwargs):
#         seats = seats.join(
#             1 / seats.groupby(level=0).size().rename("seat_weight"),
#             on='athletes_raceBoatId'
#         ).join(
#             results['Boat Type'], on='athletes_raceBoatId'
#         )
#         boat_order = results.index
#         athlete_order = athletes.index

#         weights = {
#             f: results.groupby([
#                 "raceBoats_id", f
#             ]).size().unstack(level=1, fill_value=0).loc[boat_order]
#             for f in [
#                 fields.Day,
#                 "race_event_competition_venueId",
#                 fields.race_boatClass,
#                 fields.BoatType,
#             ]
#         }
#         weights['athlete'] = seats.seat_weight.unstack(
#             level=1, fill_value=0).loc[boat_order, athlete_order]

#         Ws = {k: jnp.array(df.values) for k, df in weights.items()}
#         grams = {k: W @ W.T for k, W in Ws.items()}

#         start_times = results[fields.race_Date]
#         first_time = start_times.min()
#         last_time = start_times.max()
#         first_year = first_time.year + first_time.day_of_year / 365.25
#         last_year = last_time.year + last_time.day_of_year / 365.25
#         times = (start_times - first_time).dt.total_seconds().values
#         hours = times / 60 / 60
#         years = (last_year - first_year) * \
#             (times - times.min())/(times.max() - times.min())

#         return cls(
#             hours0=hours,
#             hours1=hours,
#             years0=years,
#             years1=years,
#             y=results.PGMT.values,
#             gram_venue=grams['race_event_competition_venueId'],
#             gram_athlete=grams['athlete'],
#             gram_boatclass=grams['Boat Class'],
#             metadata={
#                 "weights": weights,
#             },
#             **kwargs
#         )

#     def get_kernels(self):
#         boatclass_var = jnp.exp(hk.get_parameter(
#             fields.BoatType, [], init=jnp.zeros))

#         K_race_times = self.race_kernel().K(self.hours0, self.hours1) * self.gram_venue
#         K_athlete_times = self.athlete_kernel().K(
#             self.years0, self.years1) * self.gram_athlete
#         K_boatclass = boatclass_var * self.gram_boatclass
#         return (
#             K_race_times, K_athlete_times, K_boatclass,
#         )
