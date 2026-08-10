"""Geometry shared across the planes.

Lives above all of them so none has to import another: fusion pairs boxes with
the same overlap rule the evaluator scores them with, the guardrails measure
distance the same way geolocation does, and one definition means they cannot
drift apart.
"""

from pyproj import Geod

# WGS84. One instance for the whole system.
_GEOD = Geod(ellps="WGS84")


def ground_distance_m(a, b):
    """Geodesic distance in metres between two (lat, lon) points."""
    return _GEOD.inv(a[1], a[0], b[1], b[0])[2]


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0
