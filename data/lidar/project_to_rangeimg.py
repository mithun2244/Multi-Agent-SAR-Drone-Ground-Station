"""Point cloud -> 2D range image, as a grayscale PNG.

    python data/lidar/project_to_rangeimg.py --demo
    python data/lidar/project_to_rangeimg.py --input cloud.bin --output range.png
    python data/lidar/project_to_rangeimg.py --input scan.ply --width 2048 --height 64
    python data/lidar/project_to_rangeimg.py --selfcheck

The LiDAR detector in `src/perception/detectors.py` runs on a *range image*, not
on a raw cloud: a spherical projection puts the sensor at the origin, azimuth on
the x axis and elevation on the y axis, with each pixel holding the distance to
the nearest return along that ray. That turns an unordered set of points into
something a convolutional detector can read, and keeps the geometry recoverable
— pixel plus range is a bearing and a distance, which is what geolocation wants.

Nearest return wins each pixel. A far wall behind a person must never overwrite
the person: in a search, the near surface is the subject.

Readers for `.bin` (KITTI float32 x,y,z,intensity), `.ply` (ascii and binary
little-endian) and `.las` are stdlib — struct and a header parse, no laspy, no
open3d, no numpy. The PNG writer is zlib plus four chunks.

ponytail: no `.laz` (that needs a real LAZ decompressor), no `.pcd`, and the
projection is a plain spherical one — no motion compensation, no per-beam
calibration table. Real sensors ship an intrinsic table mapping beam index to
elevation, which beats an evenly-spaced fan; feed one in when the airframe's
sensor is known.
"""

import argparse
import math
import struct
import sys
import zlib
from pathlib import Path

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 64

# Vertical field of view, degrees. The default is a 64-beam scanner pointed at
# the ground from a drone: a little above the horizon, a long way below it.
DEFAULT_FOV_UP = 3.0
DEFAULT_FOV_DOWN = -25.0

EMPTY = 0        # a pixel no return landed in


# -- readers ---------------------------------------------------------------

def read_bin(path):
    """KITTI-style raw floats: x, y, z, intensity, repeating."""
    raw = Path(path).read_bytes()
    stride = 4 * 4
    if len(raw) % stride:
        # A 3-float cloud is the other common layout; anything else is a guess,
        # and guessing the stride silently produces a plausible wrong picture.
        stride = 3 * 4
        if len(raw) % stride:
            raise ValueError(f"{path}: not a multiple of 12 or 16 bytes")
    count = len(raw) // stride
    floats = struct.unpack(f"<{count * (stride // 4)}f", raw)
    step = stride // 4
    return [(floats[i], floats[i + 1], floats[i + 2]) for i in range(0, len(floats), step)]


_PLY_SIZES = {"char": "b", "uchar": "B", "int8": "b", "uint8": "B",
              "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
              "int": "i", "uint": "I", "int32": "i", "uint32": "I",
              "float": "f", "float32": "f", "double": "d", "float64": "d"}


def read_ply(path):
    """PLY vertices, ascii or binary_little_endian."""
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt, count, properties, in_vertex = None, 0, [], False
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: header never ended")
            parts = line.decode("ascii", "replace").split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties on vertices are not supported")
                properties.append((parts[1], parts[2]))
            elif parts[0] == "end_header":
                break

        names = [name for _, name in properties]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"{path}: vertices have no {axis} property")
        index = tuple(names.index(a) for a in ("x", "y", "z"))

        if fmt == "ascii":
            points = []
            for _ in range(count):
                values = fh.readline().split()
                points.append(tuple(float(values[i]) for i in index))
            return points
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format {fmt!r}")

        layout = "<" + "".join(_PLY_SIZES[kind] for kind, _ in properties)
        size = struct.calcsize(layout)
        points = []
        for _ in range(count):
            values = struct.unpack(layout, fh.read(size))
            points.append(tuple(float(values[i]) for i in index))
        return points


def read_las(path):
    """LAS 1.2-1.4, uncompressed. Coordinates are scaled int32 in the header."""
    with open(path, "rb") as fh:
        header = fh.read(375)
        if header[:4] != b"LASF":
            raise ValueError(f"{path}: not a LAS file")
        data_offset, = struct.unpack_from("<I", header, 96)
        record_length, = struct.unpack_from("<H", header, 105)
        count, = struct.unpack_from("<I", header, 107)
        scale = struct.unpack_from("<3d", header, 131)
        offset = struct.unpack_from("<3d", header, 155)
        if count == 0:                      # LAS 1.4 keeps the real count later
            count, = struct.unpack_from("<Q", header, 247)

        fh.seek(data_offset)
        points = []
        for _ in range(count):
            record = fh.read(record_length)
            if len(record) < 12:
                break
            x, y, z = struct.unpack_from("<3i", record, 0)
            points.append((x * scale[0] + offset[0],
                           y * scale[1] + offset[1],
                           z * scale[2] + offset[2]))
        return points


READERS = {".bin": read_bin, ".ply": read_ply, ".las": read_las}


def read_points(path):
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"{path.suffix or path.name}: expected one of "
                         f"{', '.join(sorted(READERS))}")
    return reader(path)


# -- projection ------------------------------------------------------------

def project(points, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
            fov_up=DEFAULT_FOV_UP, fov_down=DEFAULT_FOV_DOWN, max_range=None):
    """Spherical projection. Returns (rows of range in metres, stats).

    A pixel with no return holds 0.0 — distinct from a return at 0 m, which
    cannot happen, so "no data" never reads as "something touching the sensor".
    """
    up, down = math.radians(fov_up), math.radians(fov_down)
    span = up - down
    if span <= 0:
        raise ValueError("fov_up must be above fov_down")

    grid = [[0.0] * width for _ in range(height)]
    stats = {"points": len(points), "projected": 0, "outside_fov": 0,
             "occluded": 0, "max_range": 0.0}

    for x, y, z in points:
        distance = math.sqrt(x * x + y * y + z * z)
        if distance <= 0.0 or (max_range is not None and distance > max_range):
            stats["outside_fov"] += 1
            continue

        azimuth = math.atan2(y, x)                       # -pi..pi
        elevation = math.asin(max(-1.0, min(1.0, z / distance)))
        if not (down <= elevation <= up):
            stats["outside_fov"] += 1
            continue

        # Azimuth grows left to right; elevation is flipped so the sky is row 0,
        # which is what everyone expects to see when they open the PNG.
        col = int((0.5 * (azimuth / math.pi + 1.0)) * width)
        row = int((1.0 - (elevation - down) / span) * (height - 1))
        col = min(width - 1, max(0, col))
        row = min(height - 1, max(0, row))

        current = grid[row][col]
        if current and current <= distance:
            stats["occluded"] += 1          # something nearer already owns it
            continue
        if current:
            stats["occluded"] += 1
        grid[row][col] = distance
        stats["projected"] += 1
        stats["max_range"] = max(stats["max_range"], distance)

    return grid, stats


def to_grayscale(grid, max_range=None):
    """Range in metres -> 0..255, near = bright. Empty pixels stay black."""
    finite = [v for row in grid for v in row if v > 0.0]
    ceiling = max_range or (max(finite) if finite else 1.0)
    rows = []
    for row in grid:
        pixels = bytearray(len(row))
        for i, value in enumerate(row):
            if value <= 0.0:
                pixels[i] = EMPTY
                continue
            # Inverted so near returns are bright: a subject 30 m below the
            # drone should be the thing your eye lands on, not the horizon.
            shade = 1.0 - min(1.0, value / ceiling)
            pixels[i] = max(1, int(round(shade * 254)) + 1)
        rows.append(pixels)
    return rows


def write_png(path, rows):
    """8-bit grayscale PNG. zlib and four chunks — no imaging library."""
    height = len(rows)
    width = len(rows[0]) if height else 0
    raw = b"".join(b"\x00" + bytes(row) for row in rows)     # filter 0 per line

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return path


# -- demo ------------------------------------------------------------------

ALTITUDE_M = 30.0        # how far the demo sensor sits above the ground
DEMO_MAX_RANGE_M = 200.0


def synthetic_cloud(subjects=((70.0, 5.0), (90.0, -12.0)), beams=64, columns=2048):
    """A hillside with a couple of people on it, in sensor coordinates.

    Enough structure to prove the projection end to end without a dataset: a
    ground plane sampled by an evenly-spaced beam fan, plus a person-sized
    column of returns standing on it at each subject position.

    The numbers are tied together rather than picked separately. A beam at
    elevation θ below the horizon meets ground `ALTITUDE_M` down at a horizontal
    distance of `ALTITUDE_M / tan(-θ)`, so from 30 m up the shallowest beam that
    lands inside 200 m is about -8.5°, and the subjects sit at 70 and 90 m where
    the fan actually has returns. Place them closer and they fall below the
    -25° floor: the sensor is not looking there, and a demo that quietly drops
    its own subjects proves nothing.
    """
    points = []
    for beam in range(beams):
        elevation = math.radians(DEFAULT_FOV_DOWN
                                 + (DEFAULT_FOV_UP - DEFAULT_FOV_DOWN) * beam / (beams - 1))
        if elevation >= 0:
            continue                          # beams above the horizon see sky
        for column in range(columns):
            azimuth = -math.pi + 2 * math.pi * column / columns
            depth = ALTITUDE_M / math.tan(-elevation)
            depth *= 1.0 + 0.1 * math.sin(azimuth)     # gently undulating ground
            if depth <= 0 or depth > DEMO_MAX_RANGE_M:
                continue
            points.append((depth * math.cos(azimuth), depth * math.sin(azimuth),
                           -ALTITUDE_M))

    for distance, lateral in subjects:
        bearing = math.atan2(lateral, distance)
        ground = math.sqrt(distance * distance + lateral * lateral)
        for i in range(40):                   # 1.7 m of standing person
            z = -ALTITUDE_M + 1.7 * i / 39
            points.append((ground * math.cos(bearing), ground * math.sin(bearing), z))
    return points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, help="point cloud: .ply, .las or .bin")
    parser.add_argument("--output", type=Path, default=Path("range.png"))
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fov-up", type=float, default=DEFAULT_FOV_UP)
    parser.add_argument("--fov-down", type=float, default=DEFAULT_FOV_DOWN)
    parser.add_argument("--max-range", type=float, default=None,
                        help="metres; returns beyond this are dropped")
    parser.add_argument("--demo", action="store_true",
                        help="project a synthetic hillside instead of a file")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return selfcheck()
    if not args.demo and args.input is None:
        parser.error("give --input, or --demo for a synthetic cloud")

    if args.demo:
        points = synthetic_cloud()
        source = "synthetic hillside with 2 subjects"
    else:
        points = read_points(args.input)
        source = str(args.input)

    grid, stats = project(points, args.width, args.height,
                          args.fov_up, args.fov_down, args.max_range)
    write_png(args.output, to_grayscale(grid, args.max_range))

    filled = sum(1 for row in grid for v in row if v > 0.0)
    print(f"  source     {source}")
    print(f"  points     {stats['points']}  projected {stats['projected']}  "
          f"outside fov {stats['outside_fov']}  occluded {stats['occluded']}")
    print(f"  image      {args.width}x{args.height}  "
          f"{filled} pixel(s) filled ({100.0 * filled / (args.width * args.height):.1f}%)")
    print(f"  range      up to {stats['max_range']:.1f} m")
    print(f"  wrote      {args.output}")
    return 0


def selfcheck():
    """Known answers, offline, with no point cloud present."""
    import tempfile

    # A point straight ahead, one to the left, one below: each lands where the
    # projection geometry says it should.
    # height 5 puts the horizon exactly on the middle row rather than between
    # two of them, so the expected pixel is arithmetic and not a rounding call.
    grid, stats = project([(10.0, 0.0, 0.0)], width=8, height=5,
                          fov_up=10.0, fov_down=-10.0)
    assert stats["projected"] == 1
    assert grid[2][4] == 10.0, grid          # azimuth 0 -> centre column

    grid, _ = project([(0.0, 10.0, 0.0)], width=8, height=5,
                      fov_up=10.0, fov_down=-10.0)
    assert grid[2][6] == 10.0, grid          # 90 degrees left -> three quarters

    # Nearest return wins: the far wall must not overwrite the person.
    grid, stats = project([(50.0, 0.0, 0.0), (10.0, 0.0, 0.0)], width=8, height=5,
                          fov_up=10.0, fov_down=-10.0)
    assert grid[2][4] == 10.0, "the nearer return owns the pixel"
    assert stats["occluded"] == 1
    grid, _ = project([(10.0, 0.0, 0.0), (50.0, 0.0, 0.0)], width=8, height=5,
                      fov_up=10.0, fov_down=-10.0)
    assert grid[2][4] == 10.0, "order of arrival does not change which wins"

    # Outside the vertical field of view, and beyond max_range.
    _, stats = project([(0.0, 0.0, 10.0)], width=8, height=5,
                       fov_up=10.0, fov_down=-10.0)
    assert stats == {"points": 1, "projected": 0, "outside_fov": 1,
                     "occluded": 0, "max_range": 0.0}, stats
    _, stats = project([(10.0, 0.0, 0.0)], width=8, height=5, max_range=5.0)
    assert stats["projected"] == 0 and stats["outside_fov"] == 1

    # Shading: empty stays 0, near is brighter than far, and neither collides
    # with "no data".
    rows = to_grayscale([[0.0, 10.0, 100.0]], max_range=100.0)
    assert rows[0][0] == EMPTY == 0
    assert rows[0][1] > rows[0][2] >= 1, rows[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Every reader returns the same three points from its own format.
        expected = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        binary = tmp / "cloud.bin"
        binary.write_bytes(struct.pack("<12f", 1, 2, 3, 0.5, 4, 5, 6, 0.5, 7, 8, 9, 0.5))
        assert read_bin(binary) == expected

        ascii_ply = tmp / "cloud.ply"
        ascii_ply.write_text("ply\nformat ascii 1.0\nelement vertex 3\n"
                             "property float x\nproperty float y\nproperty float z\n"
                             "end_header\n1 2 3\n4 5 6\n7 8 9\n")
        assert read_ply(ascii_ply) == expected

        binary_ply = tmp / "binary.ply"
        binary_ply.write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
            b"property float x\nproperty float y\nproperty float z\n"
            b"property uchar intensity\nend_header\n"
            + b"".join(struct.pack("<3fB", *point, 7) for point in expected))
        assert read_ply(binary_ply) == expected

        las = tmp / "cloud.las"
        header = bytearray(375)
        header[0:4] = b"LASF"
        struct.pack_into("<I", header, 96, 375)          # offset to point data
        struct.pack_into("<H", header, 105, 12)          # record length
        struct.pack_into("<I", header, 107, 3)           # legacy point count
        struct.pack_into("<3d", header, 131, 0.001, 0.001, 0.001)
        struct.pack_into("<3d", header, 155, 0.0, 0.0, 0.0)
        las.write_bytes(bytes(header) + b"".join(
            struct.pack("<3i", int(x * 1000), int(y * 1000), int(z * 1000))
            for x, y, z in expected))
        assert [tuple(round(v, 3) for v in p) for p in read_las(las)] == expected

        # A cloud we did not write is refused rather than guessed at.
        try:
            read_points(tmp / "cloud.xyz")
            raise AssertionError("an unknown extension must be refused")
        except ValueError:
            pass

        # End to end: the demo cloud produces a real PNG with the subjects in it.
        grid, stats = project(synthetic_cloud(), width=256, height=32)
        assert stats["projected"] > 1000, stats
        out = write_png(tmp / "range.png", to_grayscale(grid))
        raw = out.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", raw[16:24]) == (256, 32)
        assert zlib.crc32(raw[12:29]) == struct.unpack(">I", raw[29:33])[0], "IHDR CRC"

    print("  ok  a point projects to the pixel its bearing says")
    print("  ok  the nearest return owns a pixel, whatever order it arrived in")
    print("  ok  outside the field of view and beyond max range are dropped, not clipped")
    print("  ok  empty pixels stay distinct from a return at any distance")
    print("  ok  .bin, .ply (ascii and binary) and .las all read the same points")
    print("  ok  an unknown extension is refused rather than guessed")
    print("  ok  the demo cloud writes a valid grayscale PNG")
    print("\n7 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
