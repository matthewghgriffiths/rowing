"""Unified data layer for the crew-speed models.

Two layers:

* :class:`RowingData` -- a plain (host-side) pandas container: IO, filtering and the
  canonical tables, plus the label<->integer-code maps used to interpret predictions.
* :class:`ModelInputs` -- a ``flax.struct`` jax pytree of pure arrays (atomic / long-format
  plus integer codes). Device-side, jittable, serialisable and **subsettable by index**, so an
  EP fit on an arbitrary slice is ``inputs.subset(mask)``. Both the simple ``PerformanceGP`` and
  the joint model build from it.

The arrays in :class:`ModelInputs` reproduce exactly what
``competition_model.PerformanceModel.from_data`` builds from pandas -- the categorical Gram
matrices are equality-outer-products of the integer codes, ``W_athlete`` is a scatter of the
seat weights, and ``year``/``hour`` are precomputed (on the full set, so subsets stay
consistent).
"""

from functools import cached_property

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from rowing.world_rowing import fields

# Categorical boat fields -> the code attribute / size attribute on ModelInputs.
_CATEGORICAL = {"venue": "n_venues", "class": "n_classes", "type": "n_types", "comp": "n_comps"}

# Senior international competition types kept by from_world_rowing_data.
SENIOR_COMPETITION_TYPES = (
    "Olympic Games",
    "Paralympics",
    "World Rowing Championships",
    "European Rowing Championships",
    "World Rowing Olympic Qualification Regatta",
    "World Rowing Olympic and Paralympic Qualification regatta",
    "World Rowing Cup I",
    "World Rowing Cup II",
    "World Rowing Cup III",
)


@flax.struct.dataclass
class ModelInputs:
    """Atomic, sliceable array view of a filtered set of race results.

    Per boat-result arrays are length ``n_boats``; per-seat arrays encode the athlete<->boat
    incidence in COO form. Categorical features are integer codes; their cardinalities are static.
    """

    y: jax.Array  # PGMT, the regression target
    year: jax.Array  # normalised year (computed on the full set)
    hour: jax.Array
    boat_venue: jax.Array
    boat_class: jax.Array
    boat_type: jax.Array
    boat_lane: jax.Array  # race-centred lane
    boat_comp: jax.Array
    seat_boat: jax.Array  # COO: boat index of each seat
    seat_athlete: jax.Array  # COO: athlete index of each seat
    seat_weight: jax.Array  # 1 / crew size
    year0: jax.Array

    n_athletes: int = flax.struct.field(pytree_node=False)
    n_venues: int = flax.struct.field(pytree_node=False)
    n_classes: int = flax.struct.field(pytree_node=False)
    n_types: int = flax.struct.field(pytree_node=False)
    n_comps: int = flax.struct.field(pytree_node=False)

    @property
    def n_boats(self):
        return self.y.shape[0]

    # --- categorical features -----------------------------------------------------------------
    def one_hot(self, field):
        codes = getattr(self, f"boat_{field}")
        n = getattr(self, _CATEGORICAL[field])
        return jax.nn.one_hot(codes, n, dtype=self.y.dtype)

    def categorical_gram(self, field):
        """Gram of the one-hot encoding: 1 where two boats share the category, else 0."""
        codes = getattr(self, f"boat_{field}")
        return (codes[:, None] == codes[None, :]).astype(self.y.dtype)

    # --- athlete incidence --------------------------------------------------------------------
    def W_athlete(self):
        W = jnp.zeros((self.n_boats, self.n_athletes), self.y.dtype)
        return W.at[self.seat_boat, self.seat_athlete].add(self.seat_weight)

    def gram_athlete(self):
        W = self.W_athlete()
        return W @ W.T

    # --- lane ---------------------------------------------------------------------------------
    @property
    def W_lane(self):
        return self.boat_lane[:, None]

    def gram_lane(self):
        return self.boat_lane[:, None] * self.boat_lane[None, :]

    # --- subsetting (the EP enabler) ----------------------------------------------------------
    def subset(self, boat_mask) -> "ModelInputs":
        """Return the inputs restricted to ``boat_mask`` (boolean or integer boat indices).

        Boats are selected directly; seats are kept where their boat survives. Categorical code
        cardinalities are preserved (codes are *not* densified) so that label<->code maps held by
        the parent :class:`RowingData` stay valid across subsets.
        """
        boat_idx = jnp.asarray(boat_mask)
        if boat_idx.dtype == bool:
            boat_idx = jnp.flatnonzero(boat_idx)

        # Map old boat index -> new position (or -1 if dropped); keep surviving seats.
        new_pos = jnp.full((self.n_boats,), -1).at[boat_idx].set(jnp.arange(boat_idx.shape[0]))
        seat_keep = new_pos[self.seat_boat] >= 0

        return self.replace(
            y=self.y[boat_idx],
            year=self.year[boat_idx],
            hour=self.hour[boat_idx],
            boat_venue=self.boat_venue[boat_idx],
            boat_class=self.boat_class[boat_idx],
            boat_type=self.boat_type[boat_idx],
            boat_lane=self.boat_lane[boat_idx],
            boat_comp=self.boat_comp[boat_idx],
            seat_boat=new_pos[self.seat_boat][seat_keep],
            seat_athlete=self.seat_athlete[seat_keep],
            seat_weight=self.seat_weight[seat_keep],
        )


class RowingData:
    """Host-side pandas container + bridge to :class:`ModelInputs`.

    Holds the canonical filtered tables (``competitions``, ``results``, ``athletes``, ``seats``)
    and, after :meth:`to_inputs`, the label<->integer-code maps needed to read predictions back
    into athlete / boat / venue / class labels.
    """

    def __init__(self, competitions, results, athletes, seats, *, entries=None):
        self.competitions = competitions
        self.results = results  # indexed by raceBoats_id; the per-boat results ("boats")
        self.athletes = athletes  # indexed by athletes_personId
        self.seats = seats
        self.entries = entries

    # label indices (set by to_inputs)
    boat_index = None
    athlete_index = None
    venue_labels = None
    class_labels = None
    type_labels = None
    comp_labels = None

    @cached_property
    def _seat_levels(self):
        """(raceBoatId, personId) for each seat, from index levels or columns."""
        s = self.seats
        if isinstance(s.index, pd.MultiIndex):
            return s.index.get_level_values(0), s.index.get_level_values(1)
        return s["athletes_raceBoatId"], s["athletes_personId"]

    def to_inputs(self) -> ModelInputs:
        r = self.results
        boat_order = r.index
        athlete_order = self.athletes.index
        self.boat_index = boat_order
        self.athlete_index = athlete_order

        venue, self.venue_labels = pd.factorize(r["race_event_competition_venueId"])
        cls, self.class_labels = pd.factorize(r[fields.race_boatClass])
        typ, self.type_labels = pd.factorize(r[fields.BoatType])
        comp_col = "race_event_competitionId" if "race_event_competitionId" in r else "race_id"
        comp, self.comp_labels = pd.factorize(r[comp_col])

        lane = (r["Lane"] - r.groupby("race_id")["Lane"].transform("mean")).to_numpy()

        start = r[fields.race_Date]
        first, last = start.min(), start.max()
        first_year = first.year + first.day_of_year / 365.25
        last_year = last.year + last.day_of_year / 365.25
        times = (start - first).dt.total_seconds().to_numpy()
        span = times.max() - times.min()
        years = first_year + (last_year - first_year) * (times - times.min()) / span
        hours = times / 60 / 60
        year0 = first_year - 2

        rb, pid = self._seat_levels
        crew_size = pd.Series(1, index=rb).groupby(rb).transform("size").to_numpy()
        seat_weight = 1.0 / crew_size
        seat_boat = boat_order.get_indexer(rb)
        seat_athlete = athlete_order.get_indexer(pid)
        keep = (seat_boat >= 0) & (seat_athlete >= 0)

        f64 = jnp.float64
        i32 = jnp.int32
        return ModelInputs(
            y=jnp.asarray(r.PGMT.to_numpy(), f64),
            year=jnp.asarray(years, f64),
            hour=jnp.asarray(hours, f64),
            boat_venue=jnp.asarray(venue, i32),
            boat_class=jnp.asarray(cls, i32),
            boat_type=jnp.asarray(typ, i32),
            boat_lane=jnp.asarray(lane, f64),
            boat_comp=jnp.asarray(comp, i32),
            seat_boat=jnp.asarray(seat_boat[keep], i32),
            seat_athlete=jnp.asarray(seat_athlete[keep], i32),
            seat_weight=jnp.asarray(seat_weight[keep], f64),
            year0=jnp.asarray(year0, f64),
            n_athletes=len(athlete_order),
            n_venues=len(self.venue_labels),
            n_classes=len(self.class_labels),
            n_types=len(self.type_labels),
            n_comps=len(self.comp_labels),
        )

    # --- IO -----------------------------------------------------------------------------------
    _CACHE_FILES = {
        "competitions": "senior_competitions.feather",
        "results": "senior_results.feather",
        "athletes": "senior_athletes.feather",
        "seats": "senior_raceBoats.feather",
    }

    @classmethod
    def from_cache(cls, dir="."):
        """Load the senior_*.feather caches written by the World Rowing Data workflow."""
        from pathlib import Path

        d = Path(dir)
        return cls(**{k: pd.read_feather(d / fn) for k, fn in cls._CACHE_FILES.items()})

    @classmethod
    def from_world_rowing_data(cls, data_dir, *, senior_types=None, phases=("Final A", "Final B", "Final C")):
        """Aggregate the per-year World Rowing API dump into the senior dataset.

        Replicates the World Rowing Data workflow: concatenate the per-year ``competitions-*`` /
        ``results-*`` / ``race_boat_athletes-*`` feathers, keep final-phase results from senior
        competition types with a finish time, and reduce to the racing athletes. Fully offline.
        """
        from pathlib import Path

        d = Path(data_dir)
        senior_types = SENIOR_COMPETITION_TYPES if senior_types is None else senior_types

        def stack(pattern, transform=None):
            frames = {int(f.stem[-4:]): pd.read_feather(f) for f in d.glob(pattern)}
            if transform:
                frames = {y: transform(df) for y, df in frames.items()}
            return pd.concat(frames, names=["year"]).sort_index()

        competitions = stack("competitions-*.feather")
        race_boat_athletes = stack("race_boat_athletes-*.feather")
        final_results = stack("results-*.feather", lambda df: df[df.Phase.isin(phases)])

        senior_competitions = (
            competitions[competitions["Competition Type"].isin(senior_types)].reset_index(0).reset_index(drop=True)
        )
        senior_results = (
            final_results[
                final_results.race_event_competition_id.isin(senior_competitions.competition_id)
                & final_results["Finish Time"].notna()
            ]
            .reset_index(0)
            .reset_index(drop=True)
        )
        senior_seats = (
            race_boat_athletes[race_boat_athletes.athletes_raceBoatId.isin(senior_results.raceBoats_id)]
            .reset_index(0)
            .reset_index(drop=True)
        )
        senior_athletes = (
            senior_seats.groupby("athletes_personId")
            .first()
            .drop(columns=["athletes_boatPosition", "athletes_raceBoatId"])
            .reset_index()
        )
        return cls(
            competitions=senior_competitions,
            results=senior_results,
            athletes=senior_athletes,
            seats=senior_seats,
        )

    def save_cache(self, dir="."):
        """Write the senior_*.feather caches (so from_cache / the notebooks see this data)."""
        from pathlib import Path

        d = Path(dir)
        for attr, fn in self._CACHE_FILES.items():
            getattr(self, attr).reset_index(drop=True).to_feather(d / fn)

    @classmethod
    def from_api(cls, years=range(2019, 2030), **kwargs):
        """Build the senior dataset from the live World Rowing API (network; user-run).

        Not wired yet -- the download is the ``World Rowing Data.ipynb`` workflow. Run that to
        refresh the ``senior_*.feather`` caches, then use :meth:`from_cache`.
        """
        raise NotImplementedError(
            "Live download is the World Rowing Data workflow; refresh the senior_*.feather caches "
            "and use RowingData.from_cache(). See world_rowing_model/World Rowing Data.ipynb."
        )

    def add_competition(self, competition_id, *, events=None, crews=None, competitors=None):
        """Attach a target competition's entry tables (boats/athletes) for prediction.

        Pass the already-fetched ``crews``/``competitors`` frames (e.g. from a cached raw-data
        workbook); live fetching via the API is the user-run World Rowing Data workflow.
        """
        if crews is None or competitors is None:
            raise NotImplementedError(
                "Pass crews=/competitors= (e.g. from the cached raw-data workbook); live API fetch "
                "is the user-run World Rowing Data workflow."
            )
        self.entries = {"events": events, "crews": crews, "competitors": competitors}
        return self

    def to_excel(self, path):
        tables = {
            "competitions": self.competitions,
            "results": self.results,
            "athletes": self.athletes,
            "seats": self.seats,
        }
        if self.entries:
            tables.update(self.entries)
        with pd.ExcelWriter(path) as xlf:
            for name, df in tables.items():
                df.to_excel(xlf, merge_cells=False, sheet_name=name)

    # --- filtering ----------------------------------------------------------------------------
    def filter(self, *, dedup_seats=True, **kwargs) -> "RowingData":
        """Return a filtered RowingData (wraps competition_model.filter_results)."""
        from rowing.model.performance import competition_model

        senior = {
            "results": self.results,
            "athletes": self.athletes,
            "seats": self.seats,
            "competitions": self.competitions,
        }
        filtered = competition_model.filter_results(senior, **kwargs)
        if dedup_seats:
            filtered["seats"] = filtered["seats"].loc[~filtered["seats"].index.duplicated()]
        return RowingData(**filtered, entries=self.entries)

    # --- model builders -----------------------------------------------------------------------
    def performance_gp(self, params=None, **kernels):
        """Build a PerformanceGP from this (filtered) data, optionally loading legacy params."""
        from rowing.model.performance.competition_model import PerformanceGP

        return PerformanceGP.from_inputs(self.to_inputs(), params=params, **kernels)

    # --- prediction ---------------------------------------------------------------------------
    def predict_competition(self, gp, boats, comp_athletes, *, start, n_samples=50_000, seed=2):
        """Predict a target competition's athlete/boat scores and per-event rank probabilities.

        Lifts the Predict_Competition workflow: athlete posterior scores -> boat scores+cov ->
        Monte-Carlo rank simulation per event. Returns a dict of pandas frames keyed back to the
        competition's labels (boat ids, athletes). ``boats``/``comp_athletes`` are the entry tables.
        """
        from scipy import stats

        from rowing.model.performance.competition_model import predict_boat_scores

        mi = self.to_inputs()
        athlete_cols = self.athlete_index
        system = gp.gp_system()
        noise = float(jnp.exp(gp.log_noise[...]))

        # best historical athlete score -> boat scores + covariance
        ascore_years = gp.predict_athletes_scores(gp.years, athletes_index=athlete_cols, system=system)
        _, cov_ath = gp.predict_athletes_score(start, system=system)
        y_boat, cov_boat = predict_boat_scores(
            ascore_years.max(axis=1).values, np.asarray(cov_ath), comp_athletes, athlete_cols, noise=noise
        )

        boatclass_var = float(gp.boatclass_var.value)
        boat_class = pd.Series(
            np.asarray(system.a @ (boatclass_var * np.asarray(mi.one_hot("class")))), index=self.class_labels
        )

        # athlete score trajectories over a year grid (for the predictions table)
        times = np.arange(self.results.year.min(), np.ceil(start) + 0.1, 0.25)
        score_year = times[times.searchsorted(start)]
        athlete_scores = gp.predict_athletes_scores(times, athletes_index=self.athlete_index, system=system)

        # Monte-Carlo finishing-rank probabilities per event
        np.random.seed(seed)
        event_ranks = {}
        for event, event_boats in boats.groupby("Event"):
            mvn = stats.multivariate_normal(
                y_boat.loc[event_boats.id].values, cov_boat.loc[event_boats.id, event_boats.id].values, allow_singular=True
            )
            ranks = (
                pd.DataFrame(stats.rankdata(-mvn.rvs(size=n_samples), axis=1), columns=event_boats.id)
                .apply(pd.Series.value_counts)
                .fillna(0)
                .T
            )
            event_ranks[event] = ranks / ranks.values.sum(1, keepdims=True)
        event_ranks = pd.concat(event_ranks, names=["event", "boatId"])
        event_ranks.columns = event_ranks.columns.astype(int)
        # Expected rank-score from P(rank=1..6); reindex so events with <6 boats don't KeyError.
        exp_score = event_ranks.reindex(columns=range(1, 7), fill_value=0).fillna(0) @ np.arange(6, 0, -1)

        return {
            "y_boat": y_boat,
            "cov_boat": cov_boat,
            "boat_class": boat_class,
            "athlete_scores": athlete_scores,
            "score_year": score_year,
            "event_ranks": event_ranks,
            "exp_score": exp_score,
        }
