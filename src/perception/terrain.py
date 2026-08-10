"""Digital elevation models — "how high is the ground at this point".

Geolocation needs terrain height to turn a camera ray into a position when no
sensor measured the range. A single assumed elevation is wrong the moment the
ground is not flat, and in mountain search-and-rescue it never is: on a 30-degree
slope, assuming flat ground puts the target tens of metres from where it is.

Four sources, all behind the same three-member interface — `elevation(lat, lon)`,
`vertical_uncertainty_m`, `resolution_m` — so the ray-marching in
`geolocation.intersect_dem` swaps between them without changing a line:

  * `ConstantDEM`   — flat ground; honest only over water or a valley floor.
  * `GridDEM`       — a lat/lon grid from arrays or JSON, for tests and demos.
  * `SrtmHgtDEM`    — **real SRTM tiles**, standard library only.
  * `GeoTiffDEM`    — **real GeoTIFF/Copernicus tiles**, via rasterio.

`.hgt` is handled without any dependency because the format is a raw
big-endian int16 square with its corner in the filename — there is nothing to
guess. GeoTIFF is a container with dozens of legal encodings, so that one
delegates to rasterio rather than shipping a half-correct parser that fails on
somebody's tile in the field.

`vertical_uncertainty_m` is not decoration: geolocation propagates it into the
reported range uncertainty, and it is what tells a consumer how much to trust an
inferred fix versus a measured one.
"""

import json
import math
import struct
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


class RasterDEM:
    """Shared sampling for a north-up raster of elevations.

    Rows run north to south, the way every DEM raster on disk is laid out, and
    `(row, col)` addresses a *sample*, not a cell corner. Bilinear between
    samples, clamped at the edge rather than extrapolated.

    Subclasses supply the pixel grid and the affine placing it on the ellipsoid;
    everything about turning a latitude and longitude into a height lives here,
    so `SrtmHgtDEM` and `GeoTiffDEM` cannot disagree about what a coordinate
    means.
    """

    # SRTM marks unfilled samples with this. Treating it as an elevation would
    # put a target 32 km underground.
    VOID = -32768

    def __init__(self, samples, origin, pixel_size, nodata=VOID,
                 vertical_uncertainty_m=_DEFAULT_VERTICAL_UNCERTAINTY_M, name=""):
        if not samples or not samples[0]:
            raise ValueError("raster is empty")
        self.samples = samples
        self.rows, self.cols = len(samples), len(samples[0])
        # (lat, lon) of the centre of the top-left sample.
        self.lat_origin, self.lon_origin = origin
        # Degrees per sample: latitude decreases going down the raster.
        self.lat_step, self.lon_step = pixel_size
        self.nodata = nodata
        self.vertical_uncertainty_m = float(vertical_uncertainty_m)
        self.name = name

    @property
    def lat_max(self):
        return self.lat_origin

    @property
    def lat_min(self):
        return self.lat_origin - (self.rows - 1) * self.lat_step

    @property
    def lon_min(self):
        return self.lon_origin

    @property
    def lon_max(self):
        return self.lon_origin + (self.cols - 1) * self.lon_step

    @property
    def resolution_m(self):
        """Post spacing in metres, at this tile's latitude.

        Ray-marching steps at this size, so it must shrink with the tile: a
        one-arcsecond tile resolves terrain a three-arcsecond one cannot.
        """
        middle = math.radians((self.lat_min + self.lat_max) / 2.0)
        return min(self.lat_step * 111_320.0,
                   self.lon_step * 111_320.0 * max(0.1, math.cos(middle)))

    def covers(self, latitude, longitude):
        return (self.lat_min <= latitude <= self.lat_max
                and self.lon_min <= longitude <= self.lon_max)

    def elevation(self, latitude, longitude):
        row = (self.lat_origin - latitude) / self.lat_step
        col = (longitude - self.lon_origin) / self.lon_step
        row = min(max(row, 0.0), self.rows - 1.0)
        col = min(max(col, 0.0), self.cols - 1.0)

        r0, c0 = int(row), int(col)
        r1, c1 = min(r0 + 1, self.rows - 1), min(c0 + 1, self.cols - 1)
        dr, dc = row - r0, col - c0

        corners = [
            (self.samples[r0][c0], (1 - dr) * (1 - dc)),
            (self.samples[r0][c1], (1 - dr) * dc),
            (self.samples[r1][c0], dr * (1 - dc)),
            (self.samples[r1][c1], dr * dc),
        ]
        # Voids do not average: interpolating a hole toward -32768 would drag a
        # real height down. Weight only the samples that are real.
        total = sum(weight for value, weight in corners if value != self.nodata)
        if total <= 0.0:
            return None
        return sum(value * weight for value, weight in corners
                   if value != self.nodata) / total

    def __repr__(self):
        return (f"{type(self).__name__}({self.rows}x{self.cols}, "
                f"lat {self.lat_min:.4f}..{self.lat_max:.4f}, "
                f"lon {self.lon_min:.4f}..{self.lon_max:.4f}, "
                f"{self.resolution_m:.0f} m posts)")


class SrtmHgtDEM(RasterDEM):
    """A real SRTM `.hgt` tile. Standard library only.

    The format is fully specified and needs no parser: a square of big-endian
    signed 16-bit heights, north-west corner encoded in the filename
    (`N46E008.hgt`), and a side length that identifies the resolution —
    3601 for one arcsecond, 1201 for three. Nothing here is guesswork, which is
    why this one ships without a dependency.
    """

    # The two sides NASA actually ships: SRTM1 (one arcsecond) and SRTM3.
    SIDES = {3601: 1.0 / 3600.0, 1201: 3.0 / 3600.0}

    def __init__(self, path, vertical_uncertainty_m=_DEFAULT_VERTICAL_UNCERTAINTY_M):
        path = Path(path)
        raw = path.read_bytes()
        side = _square_side(len(raw))
        if side not in self.SIDES:
            raise ValueError(
                f"{path.name}: {len(raw)} bytes is not a known SRTM tile "
                f"(expected a square of {sorted(self.SIDES)} int16 samples)")

        step = self.SIDES[side]
        # ">%dh" is the whole tile in one read: big-endian signed shorts.
        flat = struct.unpack(f">{side * side}h", raw)
        samples = [list(flat[r * side:(r + 1) * side]) for r in range(side)]

        south, west = parse_hgt_name(path.name)
        # Samples sit on the tile edges, so the top-left sample is at the
        # north-west corner exactly — not half a pixel inside it.
        super().__init__(samples, origin=(south + 1.0, west), pixel_size=(step, step),
                         vertical_uncertainty_m=vertical_uncertainty_m, name=path.name)


def parse_hgt_name(name):
    """`N46E008.hgt` -> (46.0, 8.0): the tile's south-west corner."""
    stem = Path(name).stem.upper()
    try:
        lat = float(stem[1:3]) * (1 if stem[0] == "N" else -1)
        lon = float(stem[4:7]) * (1 if stem[3] == "E" else -1)
    except (ValueError, IndexError):
        raise ValueError(f"{name!r} is not an SRTM tile name like 'N46E008.hgt'") from None
    if stem[0] not in "NS" or stem[3] not in "EW":
        raise ValueError(f"{name!r} is not an SRTM tile name like 'N46E008.hgt'")
    return lat, lon


class GeoTiffDEM(RasterDEM):
    """A real GeoTIFF elevation tile, read through rasterio.

    Unlike `.hgt`, GeoTIFF is a container: tiled or stripped, a dozen
    compressions, any dtype, any projection. Shipping a hand-rolled parser would
    mean one that works on the files it was tested against and fails on
    somebody's tile in the field, which for a terrain model is a silently wrong
    altitude rather than a crash. So this delegates.

    Requires a geographic (lat/lon) raster — a projected one such as UTM would
    need reprojection, and quietly treating easting as longitude would put a
    target in the wrong country.
    """

    def __init__(self, path, band=1, vertical_uncertainty_m=_DEFAULT_VERTICAL_UNCERTAINTY_M):
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "GeoTiffDEM needs rasterio: pip install rasterio. "
                "For dependency-free SRTM use SrtmHgtDEM with a .hgt tile."
            ) from e

        path = Path(path)
        with rasterio.open(path) as source:
            if source.crs is None or not source.crs.is_geographic:
                raise ValueError(
                    f"{path.name}: expected a lat/lon raster, got {source.crs}. "
                    f"Reproject to EPSG:4326 first — treating projected metres as "
                    f"degrees would place targets in the wrong country.")
            transform = source.transform
            if transform.b or transform.d:
                raise ValueError(f"{path.name}: rotated rasters are not supported")

            samples = [list(map(float, row)) for row in source.read(band)]
            nodata = source.nodata if source.nodata is not None else self.VOID
            # Rasterio's transform maps pixel *corners*; the sample sits at the
            # centre, half a pixel in.
            origin = (transform.f + transform.e / 2.0, transform.c + transform.a / 2.0)
            pixel_size = (abs(transform.e), abs(transform.a))

        super().__init__(samples, origin=origin, pixel_size=pixel_size, nodata=nodata,
                         vertical_uncertainty_m=vertical_uncertainty_m, name=path.name)


def _square_side(byte_count):
    side = int(round(math.sqrt(byte_count / 2)))
    return side if side * side * 2 == byte_count else -1


def open_dem(path, **kwargs):
    """Open whichever raster this is, by extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".hgt":
        return SrtmHgtDEM(path, **kwargs)
    if suffix in (".tif", ".tiff"):
        return GeoTiffDEM(path, **kwargs)
    if suffix == ".json":
        return load_dem(path)
    raise ValueError(f"no DEM reader for {suffix!r} ({path})")


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
