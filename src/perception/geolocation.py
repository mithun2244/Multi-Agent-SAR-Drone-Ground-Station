"""Asymmetric geolocation — turn a detection pixel into a position on the map.

Phase 2 (docs/architecture.md): "A measured range beats one inferred from the
terrain, which improves geolocation." That asymmetry is the whole point of this
module and is enforced in one place, `select_range`: a LiDAR measurement
overrides a terrain inference *unconditionally*, never by confidence comparison.
A confident guess must never outrank a measurement.

Architecture also names this the hard part: "Geolocation on a moving drone,
which depends on camera calibration, the sensor extrinsics, and terrain height,
is the hard and error-prone part of this stage." So the conventions are stated
explicitly and pinned by known-answer tests rather than left implicit.

Conventions
-----------
Camera frame : x right, y down, z along the optical axis (standard pinhole).
World frame  : ENU — east, north, up — local to the drone.
yaw_deg      : heading of the optical axis, clockwise from north.
pitch_deg    : depression, *positive looking down*. 90 is straight down.
roll_deg     : rotation about the optical axis.
altitude_m   : drone altitude on the same datum as terrain_elevation_m.

Positions are computed on the WGS84 ellipsoid with pyproj, and terrain height
comes from a DEM sampled along the ray. Neither is an approximation of the
other: a flat-earth offset and an assumed ground height both fail in exactly
the terrain where search-and-rescue happens.
"""

import math
from dataclasses import dataclass
from enum import Enum

from pyproj import Geod

from ..geometry import ground_distance_m  # noqa: F401  (re-exported: callers look for it here)
from .terrain import ConstantDEM

# WGS84. All forward/inverse geodesy goes through this one instance.
_GEOD = Geod(ellps="WGS84")


class RangeSource(str, Enum):
    MEASURED_LIDAR = "MEASURED_LIDAR"      # measured by the sensor
    INFERRED_TERRAIN = "INFERRED_TERRAIN"  # derived from an assumed ground height


@dataclass(frozen=True)
class RangeEstimate:
    range_m: float
    source: RangeSource
    sigma_m: float | None = None  # stated uncertainty, used only to break ties

    def __post_init__(self):
        if not self.range_m > 0.0:
            raise ValueError(f"range must be positive, got {self.range_m}")


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Telemetry:
    """Drone pose at the moment of capture."""

    latitude: float
    longitude: float
    altitude_m: float
    yaw_deg: float = 0.0
    pitch_deg: float = 90.0  # nadir by default
    roll_deg: float = 0.0


@dataclass(frozen=True)
class Fix:
    latitude: float
    longitude: float
    elevation_m: float
    range_m: float
    range_source: RangeSource
    sigma_m: float | None = None

    @property
    def is_measured(self):
        return self.range_source is RangeSource.MEASURED_LIDAR


def select_range(candidates):
    """Pick the range to trust. A measurement always beats an inference.

    This is the asymmetry the architecture calls for, and it is deliberately
    *not* a confidence comparison: an RGB range inferred from flat-terrain
    assumptions can look arbitrarily precise while being wrong by tens of
    metres on a slope. Uncertainty only breaks ties within the same class.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("no range candidates to choose from")

    measured = [c for c in candidates if c.source is RangeSource.MEASURED_LIDAR]
    pool = measured or candidates
    return min(pool, key=lambda c: math.inf if c.sigma_m is None else c.sigma_m)


def pixel_ray(pixel, camera, telemetry):
    """Unit vector (east, north, up) pointing from the drone through `pixel`."""
    u, v = pixel
    x = (u - camera.cx) / camera.fx
    y = (v - camera.cy) / camera.fy
    z = 1.0

    roll = math.radians(telemetry.roll_deg)
    if roll:
        x, y = x * math.cos(roll) - y * math.sin(roll), x * math.sin(roll) + y * math.cos(roll)

    # Camera axes into ENU with the drone level and facing north.
    east, north, up = x, z, -y

    pitch = math.radians(telemetry.pitch_deg)
    north, up = north * math.cos(pitch) + up * math.sin(pitch), -north * math.sin(pitch) + up * math.cos(pitch)

    yaw = math.radians(telemetry.yaw_deg)
    east, north = east * math.cos(yaw) + north * math.sin(yaw), -east * math.sin(yaw) + north * math.cos(yaw)

    norm = math.sqrt(east * east + north * north + up * up)
    return (east / norm, north / norm, up / norm)


def project(latitude, longitude, altitude_m, ray, range_m):
    """The point `range_m` along `ray`, on the WGS84 ellipsoid.

    The ray is a unit vector, so its horizontal part is the ground distance and
    its vertical part the height change. The ground distance is walked as a true
    geodesic along the ray's azimuth — not a flat-earth degree offset, which
    skews with latitude and accumulates error with distance.
    """
    east, north, up = ray
    horizontal = range_m * math.hypot(east, north)
    elevation = altitude_m + range_m * up
    if horizontal < 1e-9:  # straight down: the fix is directly below the drone
        return latitude, longitude, elevation
    azimuth = math.degrees(math.atan2(east, north))
    lon2, lat2, _ = _GEOD.fwd(longitude, latitude, azimuth, horizontal)
    return lat2, lon2, elevation


def offset_enu(latitude, longitude, east_m, north_m):
    """Move a position by a local east/north offset, along the ellipsoid."""
    distance = math.hypot(east_m, north_m)
    if distance < 1e-9:
        return latitude, longitude
    lon2, lat2, _ = _GEOD.fwd(longitude, latitude, math.degrees(math.atan2(east_m, north_m)), distance)
    return lat2, lon2


def slant_range_m(telemetry, latitude, longitude, elevation_m):
    """True 3-D distance from the drone to a known world point."""
    horizontal = ground_distance_m((telemetry.latitude, telemetry.longitude), (latitude, longitude))
    return math.hypot(horizontal, elevation_m - telemetry.altitude_m)


def world_to_pixel(latitude, longitude, elevation_m, camera, telemetry):
    """Where a known world point lands in the image, or None if behind the camera.

    The exact inverse of `pixel_ray` + `project`. Used to build geometrically
    consistent synthetic frames, and to check the camera model against itself:
    world -> pixel -> geolocate must return the world point it started from.
    """
    azimuth, _, distance = _GEOD.inv(
        telemetry.longitude, telemetry.latitude, longitude, latitude
    )
    azimuth = math.radians(azimuth)
    east, north = distance * math.sin(azimuth), distance * math.cos(azimuth)
    up = elevation_m - telemetry.altitude_m

    yaw = math.radians(telemetry.yaw_deg)
    east, north = east * math.cos(yaw) - north * math.sin(yaw), east * math.sin(yaw) + north * math.cos(yaw)

    pitch = math.radians(telemetry.pitch_deg)
    north, up = north * math.cos(pitch) - up * math.sin(pitch), north * math.sin(pitch) + up * math.cos(pitch)

    x, y, z = east, -up, north
    roll = math.radians(telemetry.roll_deg)
    if roll:
        x, y = x * math.cos(roll) + y * math.sin(roll), -x * math.sin(roll) + y * math.cos(roll)

    if z <= 1e-9:
        return None  # behind the camera
    return (camera.cx + camera.fx * x / z, camera.cy + camera.fy * y / z)


def intersect_dem(ray, telemetry, dem, max_range_m=5000.0, step_m=None, tolerance_m=1e-4):
    """Range to where the camera ray meets the terrain, or None if it never does.

    This is the RGB path: with no measured depth, range comes from where the
    line of sight actually strikes the ground. Marches outward sampling the DEM
    until the ray crosses below terrain, then bisects that bracket to
    `tolerance_m`.

    Marching against a real surface, rather than solving against a plane, is
    what makes this correct on a slope — and it is why a ray angled *upward* can
    still return a range when it runs into rising ground, which a flat-ground
    model can never report.

    Returns None when the ray reaches `max_range_m` without striking terrain
    (sky, or beyond the search area). That is a real condition, not an error:
    the caller declines to place the target rather than inventing a position.
    """
    if step_m is None:
        # Never step over terrain the DEM could have resolved. A spike narrower
        # than one post cannot be represented by the DEM in the first place.
        step_m = min(50.0, max(1.0, getattr(dem, "resolution_m", 30.0)))

    def clearance(distance):
        """Height above ground, or None where the DEM has no data.

        Real tiles have voids — steep shadowed terrain SRTM never filled. A void
        is not a height, so it cannot be interpolated across or assumed to be
        sea level; the march stops instead and the caller declines to place the
        target.
        """
        lat, lon, elevation = project(
            telemetry.latitude, telemetry.longitude, telemetry.altitude_m, ray, distance
        )
        ground = dem.elevation(lat, lon)
        return None if ground is None else elevation - ground

    start = clearance(0.0)
    if start is None or start <= 0.0:
        # Drone at or below ground: bad telemetry, wrong datum, or wrong DEM.
        # Guessing here would put a rescue team somewhere invented.
        return None

    previous = 0.0
    distance = step_m
    while distance <= max_range_m:
        gap = clearance(distance)
        if gap is None:
            return None  # marched into a void; nothing here can be trusted
        if gap <= 0.0:
            low, high = previous, distance
            while high - low > tolerance_m:
                middle = (low + high) / 2.0
                midpoint = clearance(middle)
                if midpoint is None:
                    return None
                if midpoint > 0.0:
                    low = middle
                else:
                    high = middle
            hit = (low + high) / 2.0
            return RangeEstimate(hit, RangeSource.INFERRED_TERRAIN, _terrain_sigma(ray, dem))
        previous = distance
        distance += step_m
    return None


def _terrain_sigma(ray, dem):
    """Range uncertainty implied by the DEM's height uncertainty.

    A height error maps to a range error divided by the sine of the grazing
    angle: looking straight down, a 3 m height error is a 3 m range error;
    looking along a shallow slant, the same error smears far further. This is
    the number that should keep an inferred fix from ever looking as good as a
    measured one.
    """
    grazing = max(abs(ray[2]), 0.05)
    return max(dem.vertical_uncertainty_m, dem.vertical_uncertainty_m / grazing)


def geolocate(pixel, camera, telemetry, measured_ranges=(), dem=None, **march):
    """Place a detection on the map. Returns None if no range can be had.

    `measured_ranges` are RangeEstimate values from sensors that actually
    measured depth. When any is present the DEM intersection is not even
    attempted — there is nothing for it to win, and a measurement needs no
    terrain model.
    """
    ray = pixel_ray(pixel, camera, telemetry)

    candidates = [r for r in measured_ranges if r.source is RangeSource.MEASURED_LIDAR]
    if not candidates:
        inferred = intersect_dem(ray, telemetry, dem or ConstantDEM(0.0), **march)
        if inferred is None:
            return None
        candidates = [inferred]

    chosen = select_range(candidates)
    latitude, longitude, elevation = project(
        telemetry.latitude, telemetry.longitude, telemetry.altitude_m, ray, chosen.range_m
    )

    return Fix(
        latitude=latitude,
        longitude=longitude,
        elevation_m=elevation,
        range_m=chosen.range_m,
        range_source=chosen.source,
        sigma_m=chosen.sigma_m,
    )


def geolocate_clue(clue, camera, telemetry, dem=None, lidar_sigma_m=0.5, **march):
    """Geolocate a detection clue, using its measured range when it has one.

    The reference pixel is the bottom-centre of the box — where the target meets
    the ground — because that is the point the terrain intersection is actually
    about. Using the box centre would place a standing person half their height
    beyond their feet.
    """
    x1, y1, x2, y2 = clue.detection_box()
    pixel = ((x1 + x2) / 2.0, y2)

    measured = []
    range_m = clue.agent_metadata.get("range_m")
    if range_m is not None:
        measured.append(RangeEstimate(range_m, RangeSource.MEASURED_LIDAR, lidar_sigma_m))

    return geolocate(pixel, camera, telemetry, measured, dem, **march)
