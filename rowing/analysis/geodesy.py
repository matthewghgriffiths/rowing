"""
Routines for performing geodesy

based off https://www.movable-type.co.uk/scripts/latlong-vectors.html
"""
from typing import NamedTuple

import numpy as np
from numpy import sin, cos, arctan2, sqrt, pi, radians

_AVG_EARTH_RADIUS_KM = 6371.0088


def haversine(pos1, pos2):
    return _haversine(get_rad_coords(pos1), get_rad_coords(pos2))


def haversine_km(pos1, pos2):
    theta = haversine(pos1, pos2)
    return theta * _AVG_EARTH_RADIUS_KM


def bearing(pos1, pos2):
    rad = rad_bearing(pos1, pos2)
    return (rad*180/pi + 360) % 360


class LatLon(NamedTuple):
    latitude: float
    longitude: float

    def to_rad(self):
        return RadCoords(*map(np.deg2rad, self))
    
    def set_bearing(self, bearing):
        return LatLonBear(*self, bearing)
    
    bearing = bearing 
    haversine = haversine 
    haversine_km = haversine_km 

    def follow(self, distance, bearing):
        return self.set_bearing(bearing).follow(distance)
    
    def orientate(self, pos):
        return self.set_bearing(self.bearing(pos))


class LatLonBear(NamedTuple):
    latitude: float
    longitude: float
    bearing: float

    def to_rad(self):
        return RadBearing(*map(np.deg2rad, self))
    
    def set_bearing(self, bearing):
        return self._replace(bearing=bearing)
    
    def follow(self, distance, bearing=None):
        if bearing is None:
            return follow_bearing(self, distance).to_latlon()
        return self.set_bearing(bearing)
        
    

class RadCoords(NamedTuple):
    phi: float
    lam: float

    def to_latlon(self):
        return LatLon(*map(np.rad2deg, self))
    
    def set_theta(self, theta):
        return RadBearing(*self, theta)


class RadBearing(NamedTuple):
    phi: float
    lam: float
    theta: float

    def to_latlon(self):
        return LatLonBear(*map(np.rad2deg, self))
    
    def set_theta(self, theta):
        return self._replace(theta=theta)


_Coords = (
    LatLon,
    LatLonBear,
    RadCoords,
    RadBearing,
)


def cross(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return Vector(a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)


class Vector(NamedTuple):
    x: float
    y: float
    z: float

    cross = cross  # type: ignore




def get_rad_coords(pos):
    try:
        lat, lon = pos.latitude, pos.longitude
        phi, lam = radians(lat), radians(lon)
    except AttributeError:
        try:
            phi, lam = pos.phi, pos.lam
        except AttributeError:
            phi, lam = pos

    return RadCoords(phi, lam)


def get_rad_bearing(pos):
    try:
        lat, lon, bearing = pos.latitude, pos.longitude, pos.bearing
        phi, lam, bearing = radians(lat), radians(lon), radians(bearing)
    except AttributeError:
        try:
            phi, lam, bearing = pos.phi, pos.lam, pos.theta
        except AttributeError:
            phi, lam, bearing = pos

    return RadBearing(phi, lam, bearing)


def to_n_vector(pos):
    phi, lam = get_rad_coords(pos)
    cosp = cos(phi)
    return Vector(cosp * cos(lam), cosp * sin(lam), sin(phi))


def from_n_vector(vec):
    x, y, z = vec
    phi = arctan2(z, np.sqrt(x**2 + y**2))
    lam = arctan2(y, x)
    return RadCoords(phi, lam)


def to_axis(pos):
    phi, lam, theta = get_rad_bearing(pos)
    sinp = sin(phi)
    cosp = cos(phi)
    sinl = sin(lam)
    cosl = cos(lam)
    sint = sin(theta)
    cost = cos(theta)
    x = sinl * cost - sinp * cosl * sint
    y = - cosl * cost - sinp * sinl * sint
    z = cosp * sint
    return Vector(x, y, z)


def _haversine(rad1, rad2):
    phi1, lam1 = rad1
    phi2, lam2 = rad2
    sindphi = sin((phi2 - phi1)/2)**2
    sindlam = sin((lam2 - lam1)/2)**2
    a = sindphi + cos(phi1) * cos(phi2) * sindlam
    return 2 * arctan2(sqrt(a), sqrt(1 - a))



def cdist_haversine(pos1, pos2):
    from scipy.spatial.distance import cdist
    return cdist(
        np.array(get_rad_coords(pos1)).T,
        np.array(get_rad_coords(pos2)).T,
        metric=_haversine
    )


def cdist_haversine_km(pos1, pos2):
    return cdist_haversine(pos1, pos2) * _AVG_EARTH_RADIUS_KM


def rad_bearing(pos1, pos2):
    phi1, lam1 = get_rad_coords(pos1)
    phi2, lam2 = get_rad_coords(pos2)

    y = sin(lam2 - lam1) * cos(phi2)
    x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(lam2 - lam1)
    return arctan2(y, x)



def estimate_bearing(positions, pos, tol=0.01):
    dist = haversine_km(positions, pos)
    weights = np.exp(- np.square(dist / tol)/2)
    if weights.sum() == 0:
        return np.nan
    else:
        return np.average(positions.bearing % 180, weights=weights)


def path_intersections(pos1, pos2):
    gc1s = to_axis(pos1)
    gc2s = to_axis(pos2)
    intersections = cross(gc1s, gc2s)
    return from_n_vector(intersections)


def follow_bearing(pos1, d):
    phi, lam, theta = get_rad_bearing(pos1)
    d /= _AVG_EARTH_RADIUS_KM
    phi2 = np.arcsin(
        np.sin(phi)*np.cos(d) + np.cos(phi)*np.sin(d)*np.cos(theta))
    lam2 = lam + np.arctan2(
        np.sin(theta)*np.sin(d)*np.cos(phi),
        np.cos(d)-np.sin(phi)*np.sin(phi2)
    )
    return RadBearing(phi2, lam2, theta)


def make_arrow(pos, arrowlength=0.3, arrowhead=0.1, arrowangle=20):
    start = get_rad_bearing(pos).to_latlon()
    tip = follow_bearing(start, arrowlength).to_latlon()
    p2 = follow_bearing(
        tip.set_bearing(start.bearing + 180 + arrowangle), 
        arrowhead
    ).to_latlon()
    p3 = follow_bearing(
        tip.set_bearing(start.bearing + 180 - arrowangle), 
        arrowhead
    ).to_latlon()
    # points = [tip, start, tip, p2, p3, tip]
    points = [p2, tip, start, tip, p3, p2]
    return LatLon(
        np.array([p.latitude for p in points]), 
        np.array([p.longitude for p in points])
    )


def make_arrow_base(pos, arrowlength=0.3, arrowhead=0.1, arrowangle=20, base_width=0.15):
    start = get_rad_bearing(pos).to_latlon()
    tip = follow_bearing(start, arrowlength).to_latlon()
    p2 = follow_bearing(
        tip.set_bearing(start.bearing + 180 + arrowangle), 
        arrowhead
    ).to_latlon()
    p3 = follow_bearing(
        tip.set_bearing(start.bearing + 180 - arrowangle), 
        arrowhead
    ).to_latlon()
    b1 = follow_bearing(
        start.set_bearing(start.bearing + 90), base_width
    ).to_latlon()
    b2 = follow_bearing(
        start.set_bearing(start.bearing - 90), base_width
    ).to_latlon()
    # points = [tip, start, tip, p2, p3, tip]
    points = [p2, tip, start, b1, b2, start, tip, p3, p2]
    return LatLon(
        np.array([p.latitude for p in points]), 
        np.array([p.longitude for p in points])
    )

