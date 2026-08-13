"""Mock drone feed on the real bus: telemetry -> detection clue -> coordinator.

Stands in for an airframe while the ground station is being wired up. It flies a
loiter over the demo search area every 3 seconds and puts the result through the
Phase 2 pipeline the real drone uses — stub RGB and LiDAR detections, weighted
box fusion, BoT-SORT, geolocation — so what lands on Redis Streams is a
`ClueContract` that coordinator fusion, the Scene VLM and the provenance guard
already know how to consume.

    python -m src.coordinator.mock_drone_publisher
    CASE_ID=case-mock-drone python -m src.coordinator.mock_drone_publisher
    python -m src.coordinator.mock_drone_publisher --check   # offline self-check

Point a coordinator at the same case to consume it from another terminal:

    CASE_ID=case-mock-drone REDIS_URL=redis://localhost:6379/0 python -m src.coordinator.demo

The mock pose is not decoration
-------------------------------
Gimbal pitch/yaw/roll, altitude and drone position are the *inputs* to
`geolocate`, not fields copied into the clue: the subject is projected into the
frame with the current pose, and the position published is the one the real
geolocation code recovers from that box. Nothing here emits a coordinate that
was not computed.

The airframe really flies its loiter circle, which is why `gmc` exists: at 3 s
between frames the orbit slides the frame further than a six-pixel target is
wide, and IoU association cannot survive that unaided. `--check` flies the same
orbit twice, compensated and not, and fails if the uncompensated run tracks
anything — the affine has to be what holds the track together, or it is a
decoration claiming to be a mechanism.

Battery and the onboard air temperature have no consumer, so they stay on the
status line rather than riding in `agent_metadata`. Temperature in particular is
*not* republished as a WEATHER_API clue: that tag requires the Open-Meteo
endpoint credential, and minting one here would be forging a third-party
reading — exactly what the provenance allow-list exists to refuse.
"""

import math
import os
import random
import sys
import time

from ..bus import FakeRedisStreams, RedisBus
from ..guardrails.audit import AuditLog
from ..guardrails.provenance import ProvenanceRegistry
from ..perception.agent import DetectionAgent
from ..perception.detectors import Target, lidar_stub, yolo11m_stub
from ..perception.geolocation import (
    Camera,
    Telemetry,
    geolocate,
    ground_distance_m,
    offset_enu,
    slant_range_m,
    world_to_pixel,
)
from ..perception.tracking import Affine, BoTSORT
from .blackboard import Blackboard
from .demo import DRONE_ID, _search_area_dem
from .fusion import CoordinatorFusion

DEFAULT_CASE_ID = "case-mock-drone"
INTERVAL_SECONDS = 3

# The loiter point (the marker in map_visualizer.py) and the subject we pretend
# is on the ground near it, at the demo's drone-to-subject offset.
HOME = (46.83160, 8.22760)
SUBJECT = (HOME[0] + 0.0008, HOME[1] + 0.0005)
AGL_M = 120.0

# The loiter circle the airframe actually flies. At this range a person is about
# six pixels wide and one tick of this orbit slides the frame roughly twice that,
# so IoU association cannot survive it unaided — which is what `gmc` is for.
ORBIT_RADIUS_M = 15.0
ORBIT_TICKS = 40

CAMERA = Camera(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)

# Two scene points, a fixed baseline apart, for the GMC estimate below.
_GMC_BASELINE_PX = 200.0


def pose(rng, dem, tick):
    """Drone pose this tick: a slow loiter circle, 120 m above terrain."""
    angle = 2.0 * math.pi * (tick % ORBIT_TICKS) / ORBIT_TICKS
    latitude, longitude = offset_enu(
        HOME[0], HOME[1],
        ORBIT_RADIUS_M * math.sin(angle) + rng.uniform(-0.3, 0.3),
        ORBIT_RADIUS_M * math.cos(angle) + rng.uniform(-0.3, 0.3),
    )
    return Telemetry(
        latitude=round(latitude, 7),
        longitude=round(longitude, 7),
        altitude_m=round(dem.elevation(latitude, longitude) + AGL_M + rng.uniform(-0.2, 0.2), 2),
        yaw_deg=round(15.0 + rng.uniform(-0.05, 0.05), 3),
        pitch_deg=round(55.0 + rng.uniform(-0.05, 0.05), 3),
        roll_deg=round(rng.uniform(-0.05, 0.05), 3),
    )


def gmc(dem, previous, current):
    """Camera motion between two poses, as frame registration would recover it.

    Two ground points a fixed baseline apart are placed under the previous pose
    and re-projected under the current one. How that image vector shifts, turns
    and stretches is the affine BoT-SORT warps its predictions by, so drone
    motion is not read as target motion.

    It uses scene geometry and telemetry only — never the target's position —
    because the moment this consults the thing being tracked it stops simulating
    registration and starts feeding the tracker the answer.

    ponytail: telemetry-derived, so it inherits whatever the pose and the DEM get
    wrong; the real estimator registers consecutive frames (ORB + RANSAC, or
    ECC), which is the Phase 2 deferral. Returns None when either reference point
    misses the terrain — above the horizon, off the DEM — and the tracker then
    runs uncompensated exactly as it does today.
    """
    if previous is None:
        return None

    before = ((CAMERA.cx, CAMERA.cy), (CAMERA.cx + _GMC_BASELINE_PX, CAMERA.cy))
    fixes = [geolocate(pixel, CAMERA, previous, dem=dem) for pixel in before]
    if any(fix is None for fix in fixes):
        return None
    after = [world_to_pixel(fix.latitude, fix.longitude, fix.elevation_m, CAMERA, current)
             for fix in fixes]
    if any(pixel is None for pixel in after):
        return None

    (p1x, p1y), (p2x, p2y) = before
    (q1x, q1y), (q2x, q2y) = after
    before_dx, before_dy = p2x - p1x, p2y - p1y
    after_dx, after_dy = q2x - q1x, q2y - q1y
    baseline = math.hypot(before_dx, before_dy)
    stretched = math.hypot(after_dx, after_dy)
    if baseline < 1e-9 or stretched < 1e-9:
        return None

    scale = stretched / baseline
    rotation = math.atan2(after_dy, after_dx) - math.atan2(before_dy, before_dx)
    # Translation is whatever is left once rotation and scale have been applied
    # to the first reference point.
    a, b = math.cos(rotation) * scale, -math.sin(rotation) * scale
    c, d = math.sin(rotation) * scale, math.cos(rotation) * scale
    return Affine.from_camera_delta(
        dx_px=q1x - (a * p1x + b * p1y),
        dy_px=q1y - (c * p1x + d * p1y),
        rotation_rad=rotation,
        scale=scale,
    )


def framed_subject(dem, telemetry, subject=SUBJECT):
    """The subject projected into this frame, or None when it is out of view.

    Box, world position and range all describe the same point, so the fix the
    agent computes from the box means something instead of measuring the stub.
    """
    ground = dem.elevation(*subject)
    feet = world_to_pixel(subject[0], subject[1], ground, CAMERA, telemetry)
    head = world_to_pixel(subject[0], subject[1], ground + 1.7, CAMERA, telemetry)
    if feet is None or head is None:
        return None
    half_width = abs(feet[1] - head[1]) * 0.35
    return Target(
        box=(feet[0] - half_width, head[1], feet[0] + half_width, feet[1]),
        geo=subject,
        range_m=slant_range_m(telemetry, subject[0], subject[1], ground),
    )


SENSORS = ("rgb", "lidar")


def run(bus, case_id, fusion=None, interval=INTERVAL_SECONDS, ticks=None, seed=7,
        use_gmc=True, quiet=False, sensors=SENSORS):
    """Fly the loiter, publishing what the pipeline confirms. Returns the agent.

    `ticks=None` flies until interrupted. A `fusion` consumes each tick's clues
    into a picture, so one terminal shows the whole chain. `use_gmc=False` flies
    the same orbit with the tracker uncompensated, which is how the self-check
    shows the compensation is doing the work. `sensors` is what is fitted to the
    airframe — one entry models the single-sensor DJI, and no fusion runs.
    """
    dem = _search_area_dem()
    # No false alarms: the point here is a clean feed to develop consumers
    # against. Raise fp_per_frame to exercise the guards instead.
    fitted = {
        "rgb": yolo11m_stub(seed=seed, recall=0.95, fp_per_frame=0.0, device_id=DRONE_ID),
        "lidar": lidar_stub(seed=seed, recall=0.85, fp_per_frame=0.0, device_id=DRONE_ID),
    }
    fitted = {name: fitted[name].detect for name in sensors}
    agent = DetectionAgent(bus, case_id, CAMERA, dem=dem, device_id=DRONE_ID,
                           detectors=fitted, tracker=BoTSORT(**_tracker_for(sensors)))

    rng = random.Random(seed)
    battery, tick, previous = 100.0, 0, None
    while ticks is None or tick < ticks:
        telemetry = pose(rng, dem, tick)
        motion = gmc(dem, previous, telemetry) if use_gmc else None
        previous = telemetry
        target = framed_subject(dem, telemetry)
        frame_id = f"frame_{tick:04d}"
        # Every fitted sensor delivers this frame; an empty list is a feed that
        # saw nothing, which is not the same as a sensor that is not there.
        seen = [target] if target else []
        published = agent.capture(frame_id, {name: seen for name in fitted}, telemetry,
                                  camera_motion=motion)

        air_c = round(rng.uniform(-4.0, 6.0), 1)
        if not quiet:
            print(_status(frame_id, telemetry, battery, air_c, motion, published))
            if fusion is not None:
                print(f"        {_picture(fusion.refresh(case_id))}")
        elif fusion is not None:
            fusion.refresh(case_id)

        battery = max(0.0, battery - 0.2)
        tick += 1
        if ticks is None or tick < ticks:
            time.sleep(interval)
    return agent


def _tracker_for(sensors):
    """Tracker settings for this sensor configuration.

    ponytail: `track_new_thresh` ships at 0.6, a number calibrated on the *fused*
    stream. The LiDAR detector is deliberately the less certain of the two, so
    alone it almost never clears 0.6 and confirms nothing — see the check in
    `test_perception.py`. Dropping it here keeps the single-sensor demo honest;
    the real number comes from a Phase 9 tuning run per airframe configuration,
    not from this file.
    """
    settings = {"min_hits": 3}
    if tuple(sensors) == ("lidar",):
        settings["new_track_thresh"] = 0.45
    return settings


def _status(frame_id, telemetry, battery, air_c, motion, published):
    gmc_note = (f"gmc {motion.tx:+6.1f},{motion.ty:+6.1f}px" if motion is not None
                else "gmc" + " " * 14)
    if published:
        clue, spatial = published[0], published[0].spatial_context
        saw = "+".join(s.replace("DRONE_", "").lower() for s in clue.agent_metadata["sensors"])
        where = (f"clue {spatial.latitude:.5f}, {spatial.longitude:.5f} "
                 f"({clue.agent_metadata['range_source']}, {saw} {clue.confidence_score:.2f})"
                 if spatial.latitude is not None else f"clue NO FIX ({saw})")
    else:
        where = "track building"
    return (
        f"{frame_id}  {telemetry.latitude:.6f}, {telemetry.longitude:.6f}  "
        f"alt {telemetry.altitude_m:7.2f}m  gimbal p{telemetry.pitch_deg:6.2f} "
        f"y{telemetry.yaw_deg:6.2f}  bat {battery:5.1f}%  {air_c:5.1f}C  "
        f"{gmc_note}  ->  {where}"
    )


def _picture(picture):
    if not picture.targets:
        return "picture: nothing detected yet"
    top = picture.targets[0]
    where = f"{top.latitude:.5f}, {top.longitude:.5f}" if top.located else "NOT LOCATED"
    return (f"picture: {len(picture.targets)} target(s), top {top.target_id} "
            f"conf {top.confidence:.3f} prio {top.priority:.3f} at {where}")


def _fly(ticks, use_gmc, quiet=False, sensors=SENSORS):
    """One offline flight, wired to a coordinator. Returns (agent, fusion, case_id)."""
    bus = RedisBus(FakeRedisStreams())
    blackboard = Blackboard()
    case = blackboard.open_case(case_id="case-check", sector="mock")
    fusion = CoordinatorFusion(
        bus, blackboard,
        provenance=ProvenanceRegistry(devices={DRONE_ID}), audit=AuditLog(),
    )
    agent = run(bus, case.case_id, interval=0.0, ticks=ticks, use_gmc=use_gmc, quiet=quiet,
                sensors=sensors)
    return agent, fusion, case.case_id


def check():
    """A published clue must clear the provenance guard and reach the picture.

    The same orbit is then flown with the compensation off. If the tracker
    confirms either way, the affine is decorative and this file is lying about
    what it does.
    """
    agent, fusion, case_id = _fly(ticks=8, use_gmc=True)
    picture = fusion.refresh(case_id)

    assert agent.published >= 1, "nothing reached the bus"
    assert not fusion.rejected, f"the guard refused our own feed: {fusion.rejected}"
    assert picture.targets, "clues published but no target in the picture"
    top = picture.targets[0]
    assert top.located, "a target reached the picture with no computed position"
    error = ground_distance_m((top.latitude, top.longitude), SUBJECT)
    assert error < 60.0, f"fix is {error:.0f} m from the simulated subject"

    uncompensated, _, _ = _fly(ticks=8, use_gmc=False, quiet=True)
    assert uncompensated.published == 0, (
        f"the orbit is trackable without GMC ({uncompensated.published} clue(s)) — "
        f"the compensation is not what is holding the track together"
    )

    # The same flight on a single-sensor airframe: no fusion, and the clue says
    # which sensor is behind it.
    singles = {}
    for sensor in SENSORS:
        alone, single_fusion, single_case = _fly(ticks=8, use_gmc=True, quiet=True,
                                                 sensors=(sensor,))
        clues = [c for _, c in _stream(single_fusion, single_case)]
        assert clues, f"a {sensor}-only airframe published nothing"
        assert not single_fusion.rejected, f"{sensor}-only feed refused: {single_fusion.rejected}"
        for clue in clues:
            assert clue.agent_metadata["sensors"] == [f"DRONE_{sensor.upper()}"], (
                f"a {sensor}-only clue claims {clue.agent_metadata['sensors']}"
            )
        singles[sensor] = len(clues)

    print(f"\nok - {agent.published} clue(s) published, {len(picture.targets)} target(s), "
          f"fix {error:.1f} m from the simulated subject; "
          f"{uncompensated.published} published with GMC off; "
          + ", ".join(f"{n} on {s} alone" for s, n in singles.items()))


def _stream(fusion, case_id):
    return fusion.bus.read(f"clues:{case_id}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        # What is fitted to the airframe. One sensor is the DJI case: only that
        # model runs and there is nothing to fuse.
        sensors = SENSORS
        if "--rgb-only" in sys.argv:
            sensors = ("rgb",)
        elif "--lidar-only" in sys.argv:
            sensors = ("lidar",)

        case_id = os.environ.get("CASE_ID", DEFAULT_CASE_ID)
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        bus = RedisBus.from_url(url)
        bus.client.ping()  # fail here, clearly, rather than publishing into nothing

        # A local coordinator so one terminal shows the whole chain. Another
        # process reading the same stream is an independent consumer — that is
        # what Redis Streams is for, and this one does not steal its clues.
        blackboard = Blackboard()
        case = blackboard.open_case(case_id=case_id, sector="mock drone loiter")
        fusion = CoordinatorFusion(
            bus, blackboard,
            provenance=ProvenanceRegistry(devices={DRONE_ID}), audit=AuditLog(),
        )

        print(f"\n  redis:  {url}")
        print(f"  case:   {case_id}  ->  stream clues:{case_id}")
        print(f"  drone:  {DRONE_ID} loitering at {HOME[0]:.5f}, {HOME[1]:.5f}")
        print(f"  fitted: {', '.join(sensors)}"
              f"{' (WBF runs)' if len(sensors) > 1 else ' (single sensor, no fusion)'}")
        print(f"  every {INTERVAL_SECONDS}s - ctrl-c to stop\n")
        try:
            run(bus, case_id, fusion=fusion, sensors=sensors)
        except KeyboardInterrupt:
            print("\nstopped")
