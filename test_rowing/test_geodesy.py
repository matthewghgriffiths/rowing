"""Unit tests for rowing.analysis.geodesy (pure spherical-geometry helpers)."""

import numpy as np
import pytest

from rowing.analysis import geodesy as g
from rowing.analysis.geodesy import LatLon, RadCoords


def test_haversine_identical_points_is_zero():
    assert g.haversine(LatLon(52.0, 0.1), LatLon(52.0, 0.1)) == 0.0


def test_haversine_is_symmetric():
    a, b = LatLon(10.0, 20.0), LatLon(30.0, 40.0)
    assert g.haversine(a, b) == pytest.approx(g.haversine(b, a))


def test_haversine_km_known_distance():
    # One degree of arc at the equator is ~111.195 km for the mean Earth radius.
    assert g.haversine_km(LatLon(0.0, 0.0), LatLon(0.0, 1.0)) == pytest.approx(111.195, abs=1e-2)
    assert g.haversine_km(LatLon(0.0, 0.0), LatLon(1.0, 0.0)) == pytest.approx(111.195, abs=1e-2)


@pytest.mark.parametrize(
    "dest, expected",
    [
        (LatLon(1.0, 0.0), 0.0),  # north
        (LatLon(0.0, 1.0), 90.0),  # east
        (LatLon(-1.0, 0.0), 180.0),  # south
        (LatLon(0.0, -1.0), 270.0),  # west
    ],
)
def test_bearing_cardinal_directions(dest, expected):
    assert g.bearing(LatLon(0.0, 0.0), dest) == pytest.approx(expected, abs=1e-6)


def test_get_rad_coords_accepts_latlon_radcoords_and_tuple():
    # LatLon is interpreted as degrees and converted to radians.
    assert g.get_rad_coords(LatLon(180.0, 90.0)) == pytest.approx((np.pi, np.pi / 2))
    # RadCoords passes through unchanged.
    assert g.get_rad_coords(RadCoords(0.1, 0.2)) == pytest.approx((0.1, 0.2))
    # A bare tuple is treated as (phi, lam) already in radians.
    assert g.get_rad_coords((0.1, 0.2)) == pytest.approx((0.1, 0.2))


def test_n_vector_roundtrip():
    pos = LatLon(52.0, 13.0)
    back = g.from_n_vector(g.to_n_vector(pos)).to_latlon()
    assert (back.latitude, back.longitude) == pytest.approx((52.0, 13.0))


def test_follow_bearing_covers_distance_and_heading():
    start = LatLon(52.0, 0.0)
    end = start.follow(5.0, 90.0)  # 5 km due east
    assert g.haversine_km(start, end) == pytest.approx(5.0, abs=1e-6)
    assert g.bearing(start, end) == pytest.approx(90.0, abs=1e-3)
