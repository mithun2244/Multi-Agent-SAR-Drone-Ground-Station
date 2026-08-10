"""Digital elevation models — "how high is the ground at this point".

Geolocation needs terrain height to turn a camera ray into a position when no
sensor measured the range. A single assumed elevation is wrong the moment the
ground is not flat, and in mountain search-and-rescue it never is: on a 30-degree
slope, assuming flat ground puts the target tens of metres from where it is.

The DEM sources here are placeholders — a constant, and a lat/lon grid you can
build from arrays or a JSON file. Both expose the same `elevation(lat, lon)`, so
swapping in a real raster (GeoTIFF via rasterio, or an SRTM/Copernicus tile) is
a new class, not a change to the geolocation code that consumes it.

`vertical_uncertainty_m` is not decoration: geolocation propagates it into the
reported range uncertainty, and it is what tells a consumer how much to trust an
inferred fix versus a measured one.
"""

import json
from pathlib import Path

# SRTM-class global DEMs are roughly this good in open terrain, worse under
# canopy and on steep slopes.
_DEFAULT_VERTICAL_UNCERTAINTY_M = 3.0


class ConstantDEM:
    """Flat ground everywhere. Honest only over water or a valley floor."""

    def __init__(self, elevation_m=0.0, vertical_uncertainty_m=_DEFAULT_VERTICAL_UNCERTAINTY_M):
        self.elevation_m = float(elevation_m)
        self.vertical_uncertainty_m = float(vertical_uncertainty_m)
        # Nothing to miss between samples on a flat surface.
        self.resolution_m = float("inf")

    def elevation(self, latitude, longitude):
        return self.elevation_m

    def __repr__(self):
        return f"ConstantDEM({self.elevation_m} m)"


class GridDEM:
    """Regular latitude/longitude grid, bilinearly interpolated.

    `rows[i][j]` is the elevation at `lat_min + i*lat_step`,
    `lon_min + j*lon_step`, so row 0 is the southern edge.

    Queries outside the tile clamp to the edge rather than extrapolating. A
    wrong-but-bounded elevation beats a linear extrapolation that runs away, and
    `covers()` lets a caller check first rather than trusting a clamped value.
    """

    def __init__(self, rows, lat_min, lon_min, lat_step, lon_step,
                 vertical_uncertainty_m=_DEFAULT_VERTICAL_UNCERTAINTY_M):
        if not rows or not rows[0]:
            raise ValueError("DEM grid is empty")
        if len({len(r) for r in rows}) != 1:
            raise ValueError("DEM grid rows must all be the same length")
        if lat_step <= 0 or lon_step <= 0:
            raise ValueError("DEM steps must be positive")

        self.rows = [list(map(float, r)) for r in rows]
        self.lat_min, self.lon_min = float(lat_min), float(lon_min)
        self.lat_step, self.lon_step = float(lat_step), float(lon_step)
        self.n_lat, self.n_lon = len(self.rows), len(self.rows[0])
        self.vertical_uncertainty_m = float(vertical_uncertainty_m)

    @property
    def lat_max(self):
        return self.lat_min + (self.n_lat - 1) * self.lat_step

    @property
    def lon_max(self):
        return self.lon_min + (self.n_lon - 1) * self.lon_step

    @property
    def resolution_m(self):
        """Post spacing in metres — the scale of terrain this DEM can resolve."""
        return min(self.lat_step * 111_320.0, self.lon_step * 111_320.0)

    def covers(self, latitude, longitude):
        return (self.lat_min <= latitude <= self.lat_max
                and self.lon_min <= longitude <= self.lon_max)

    def elevation(self, latitude, longitude):
        fi = (latitude - self.lat_min) / self.lat_step
        fj = (longitude - self.lon_min) / self.lon_step
        fi = min(max(fi, 0.0), self.n_lat - 1.0)
        fj = min(max(fj, 0.0), self.n_lon - 1.0)

        i0, j0 = int(fi), int(fj)
        i1, j1 = min(i0 + 1, self.n_lat - 1), min(j0 + 1, self.n_lon - 1)
        di, dj = fi - i0, fj - j0

        south = self.rows[i0][j0] * (1 - dj) + self.rows[i0][j1] * dj
        north = self.rows[i1][j0] * (1 - dj) + self.rows[i1][j1] * dj
        return south * (1 - di) + north * di

    @classmethod
    def from_function(cls, fn, lat_min, lon_min, lat_step, lon_step, n_lat, n_lon, **kwargs):
        """Synthesize a DEM from `fn(lat, lon) -> elevation`, for tests and demos."""
        rows = [
            [fn(lat_min + i * lat_step, lon_min + j * lon_step) for j in range(n_lon)]
            for i in range(n_lat)
        ]
        return cls(rows, lat_min, lon_min, lat_step, lon_step, **kwargs)

    def to_json(self, path):
        Path(path).write_text(json.dumps({
            "lat_min": self.lat_min, "lon_min": self.lon_min,
            "lat_step": self.lat_step, "lon_step": self.lon_step,
            "vertical_uncertainty_m": self.vertical_uncertainty_m,
            "rows": self.rows,
        }), encoding="utf-8")

    def __repr__(self):
        return (f"GridDEM({self.n_lat}x{self.n_lon}, "
                f"lat {self.lat_min:.4f}..{self.lat_max:.4f}, "
                f"lon {self.lon_min:.4f}..{self.lon_max:.4f})")


def load_dem(path):
    """Placeholder loader: the JSON form written by `GridDEM.to_json`.

    A real deployment replaces this with a raster read (rasterio/GDAL) over an
    SRTM or Copernicus tile covering the search area. Everything downstream only
    needs `elevation(lat, lon)`, `vertical_uncertainty_m`, and `resolution_m`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GridDEM(
        rows=data["rows"],
        lat_min=data["lat_min"],
        lon_min=data["lon_min"],
        lat_step=data["lat_step"],
        lon_step=data["lon_step"],
        vertical_uncertainty_m=data.get("vertical_uncertainty_m", _DEFAULT_VERTICAL_UNCERTAINTY_M),
    )
