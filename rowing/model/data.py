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
import pandas as pd

from rowing.world_rowing import fields

# Categorical boat fields -> the code attribute / size attribute on ModelInputs.
_CATEGORICAL = {"venue": "n_venues", "class": "n_classes", "type": "n_types", "comp": "n_comps"}


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
