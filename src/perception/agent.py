"""The detection agent — the Phase 2 vertical slice, end to end.

    fitted sensors -> [weighted box fusion] -> BoT-SORT -> geolocation -> bus

Architecture (Phase 2): "the agent emits one clue per confirmed track to the
bus." Only confirmed tracks are published: a single-frame blip is not a target,
and publishing one would put a phantom on the operator's map.

Sensor-agnostic
---------------
The airframe decides what runs. A drone with only a camera fitted runs only the
RGB detector; one with only a LiDAR runs only that; Weighted Box Fusion is the
step that happens *when there are two feeds to reconcile*, not a mandatory stage.
Everything after the detectors takes boxes and does not care where they came
from, so tracking and the geodesic ray march are identical in all three cases.

An absent feed and a quiet one are different things. A sensor that is fitted and
delivered a frame with nothing in it still counts as a feed, so fusion goes on
charging the other sensor's lone detection for the corroboration it did not get;
a sensor that is not fitted, or whose feed dropped this frame, is simply not
there to corroborate and nothing is deducted for its silence.

Published clues name the sensors behind them in `agent_metadata["sensors"]`, so
a consumer can tell an RGB-only sighting from one two sensors agreed on.

On a track that cannot be geolocated
------------------------------------
The clue is still published, with `spatial_context.bounding_box` set and
latitude/longitude left `None`. Two things are deliberately not done:

  * no placeholder or assumed coordinates are ever emitted — a wrong position
    sends a ground team to the wrong valley, which is worse than no position;
  * the detection is not silently dropped — "we can see someone but cannot
    place them" is real information in a search, and losing it because the DEM
    did not cover the area would be its own failure.

`agent_metadata["geolocation"]` says which case a clue is, so a consumer can
tell "not located" from "located here" without guessing from null fields.
"""

import uuid

from ..bus import stream_for
from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.provenance import TAG_TRACK
from .fusion import weighted_box_fusion
from .geolocation import RangeEstimate, RangeSource, geolocate
from .tracking import BoTSORT

TRACK_PROVENANCE = TAG_TRACK


class DetectionAgent:
    """Turns raw sensor detections into geolocated, tracked clues on the bus."""

    def __init__(
        self,
        bus,
        case_id,
        camera,
        dem=None,
        tracker=None,
        fusion_weights=None,
        fusion_iou_threshold=0.55,
        lidar_sigma_m=0.5,
        stream=None,
        device_id=None,
        detectors=None,
    ):
        self.bus = bus
        self.case_id = case_id
        self.camera = camera
        self.dem = dem
        self.tracker = tracker or BoTSORT()
        self.fusion_weights = fusion_weights
        self.fusion_iou_threshold = fusion_iou_threshold
        self.lidar_sigma_m = lidar_sigma_m
        self.stream = stream or stream_for(case_id)
        # The airframe this agent is flying on, checked by the provenance guard.
        self.device_id = device_id
        # What is fitted to that airframe: {sensor -> callable(frame_id, feed,
        # case_id) -> [ClueContract]}. Injected rather than imported by name, so
        # a real model replaces a stub without touching this file. Leave it empty
        # and call `process_frame` directly if the caller runs its own detectors.
        #
        # ponytail: `fusion_weights` stays positional, aligned to this mapping's
        # order. Two sensors and a dropped feed means one group and no fusion, so
        # the misalignment cannot bite; key the weights by sensor name if a third
        # sensor is ever fitted.
        self.detectors = dict(detectors or {})

        self.published = 0
        self.without_fix = 0
        self._last_timestamp = None

    def capture(self, frame_id, feeds, telemetry, camera_motion=None):
        """Detect on whichever fitted sensors delivered this frame, then track.

        `feeds` maps sensor name to whatever that sensor produced this frame — an
        RGB image, a LiDAR range image, the simulator's truth targets. Only
        detectors with a feed run, so a drone with no LiDAR fitted never pays for
        a LiDAR model and a feed that dropped out is not invented.
        """
        groups = [self.detectors[name](frame_id, feeds[name], self.case_id)
                  for name in self.detectors if feeds.get(name) is not None]
        return self.process_frame(frame_id, groups, telemetry, camera_motion)

    def process_frame(self, frame_id, sensor_clues, telemetry, camera_motion=None):
        """Run one frame through the pipeline. Returns the clues published.

        `sensor_clues` is one list of detections per feed carrying this frame:
        `[rgb, lidar]` on a two-sensor airframe, `[rgb]` on a DJI that only has
        a camera. Fusion runs only when there is more than one feed — with a
        single sensor there is nothing to reconcile and nothing to weigh, so its
        boxes reach the tracker untouched rather than through a one-input merge
        that could only round them.
        """
        feeds = [list(group) for group in sensor_clues]
        # Checked on the way in, not on the fused output: a foreign clue must be
        # refused whether or not fusion is the stage that would have carried it.
        for clue in (c for group in feeds for c in group):
            if clue.case_id != self.case_id:
                raise ValueError(
                    f"clue {clue.clue_id} belongs to {clue.case_id}, not {self.case_id}"
                )

        if len(feeds) > 1:
            detections = weighted_box_fusion(
                feeds,
                weights=self.fusion_weights,
                iou_threshold=self.fusion_iou_threshold,
            )
        else:
            detections = feeds[0] if feeds else []

        if detections:
            self._last_timestamp = max(c.timestamp for c in detections)

        # Which sensors are behind each detection, resolved before tracking: the
        # tracker keeps clue ids, and this is the only place that still knows
        # what those ids came off.
        origins = {clue.clue_id: _sensors_of(clue) for clue in detections}
        confirmed = self.tracker.update(detections, frame_id=frame_id,
                                        camera_motion=camera_motion)

        published = []
        for track in confirmed:
            clue = self._track_clue(track, frame_id, telemetry, origins)
            self.bus.publish(clue, stream=self.stream)
            published.append(clue)
            self.published += 1
        return published

    def _track_clue(self, track, frame_id, telemetry, origins=None):
        x1, y1, x2, y2 = track.box
        fix = geolocate(
            ((x1 + x2) / 2.0, y2),  # ground contact point, not the box centre
            self.camera,
            telemetry,
            measured_ranges=self._measured(track),
            dem=self.dem,
        )
        if fix is None:
            self.without_fix += 1

        label = track.class_label or "contact"
        # The sensors behind the detection that updated this track *this* frame,
        # not the whole frame's sensors: with two targets and only one of them in
        # the RGB frame, a frame-level list would over-claim for the other.
        sensors = (origins or {}).get(track.clue_ids[-1], []) if track.clue_ids else []
        metadata = {
            "track_id": track.track_id,
            "track_state": track.state.value,
            "hits": track.hits,
            "frames_seen": track.frames_seen,
            "first_frame_id": track.first_frame_id,
            "geolocation": "located" if fix else "no_fix",
            "sensors": list(sensors),
        }
        if self.device_id:
            metadata["device_id"] = self.device_id
        if fix is not None:
            metadata["range_m"] = round(fix.range_m, 3)
            metadata["range_source"] = fix.range_source.value
            metadata["range_sigma_m"] = None if fix.sigma_m is None else round(fix.sigma_m, 3)
        elif track.range_m is not None:
            metadata["range_m"] = track.range_m

        return ClueContract(
            # Deterministic in (case, track, frame): replaying a frame updates
            # the same entry rather than duplicating the target.
            clue_id=str(uuid.uuid5(
                uuid.NAMESPACE_URL, f"sar:track:{self.case_id}:{track.track_id}:{frame_id}"
            )),
            case_id=self.case_id,
            parent_clue_ids=list(track.clue_ids),
            timestamp=self._last_timestamp,
            source_agent=AgentSource.PERCEPTION_FUSION,
            confidence_score=track.confidence,
            finding_summary=self._summary(track, label, fix, sensors),
            spatial_context=SpatialContext(
                latitude=fix.latitude if fix else None,
                longitude=fix.longitude if fix else None,
                altitude_m=fix.elevation_m if fix else None,
                bounding_box=[round(v, 2) for v in track.box],
            ),
            frame_id=frame_id,
            class_label=track.class_label,
            provenance_tag=TRACK_PROVENANCE,
            agent_metadata=metadata,
        )

    def _measured(self, track):
        if track.range_m is None:
            return ()
        return (RangeEstimate(track.range_m, RangeSource.MEASURED_LIDAR, self.lidar_sigma_m),)

    @staticmethod
    def _summary(track, label, fix, sensors=()):
        seen = f"track {track.track_id}, {track.frames_seen} frames"
        if sensors:
            seen += f", {'+'.join(sensors)}"
        if fix is None:
            return f"Confirmed {label} ({seen}); position unavailable, no range to terrain"
        how = "measured LiDAR range" if fix.is_measured else "range inferred from terrain"
        return (
            f"Confirmed {label} ({seen}) at {fix.latitude:.6f}, {fix.longitude:.6f}, "
            f"{fix.range_m:.1f} m away by {how}"
        )


def _sensors_of(clue):
    """Which physical sensors are behind one detection.

    A fused clue names its parents; a single-sensor detection is its own source.
    Either way the published track says which sensor actually saw the target,
    which is the difference between "two sensors agree" and "the camera thinks
    so" — and on a single-sensor airframe that is the whole story.
    """
    fused_from = clue.agent_metadata.get("wbf_sources")
    return list(fused_from) if fused_from else [clue.source_agent.value]


if __name__ == "__main__":  # a full frame-by-frame run, for eyeballing the bus
    import os

    from ..bus import FakeRedisStreams, RedisBus
    from .detectors import Target, lidar_stub, yolo11m_stub
    from .geolocation import Camera, Telemetry, ground_distance_m, slant_range_m, world_to_pixel
    from .terrain import GridDEM

    CASE = "case-2026-0042"
    DRONE = (46.8182, 8.2275)

    # An alpine slope rising ~180 m per km toward the north.
    dem = GridDEM.from_function(
        lambda lat, lon: 1450.0 + (lat - DRONE[0]) * 20_000.0,
        lat_min=46.79, lon_min=8.20, lat_step=0.0005, lon_step=0.0005,
        n_lat=120, n_lon=120,
    )
    camera = Camera(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    telemetry = Telemetry(DRONE[0], DRONE[1], altitude_m=1570.0, yaw_deg=15.0, pitch_deg=55.0)

    # Project two real world positions into the frame, so the boxes, the world
    # positions and the ranges all describe the same targets. Anything else and
    # "distance from truth" below would measure the stub, not the geolocation.
    truth = []
    for lat, lon in ((46.8190, 8.2280), (46.8186, 8.2265)):
        ground = dem.elevation(lat, lon)
        feet = world_to_pixel(lat, lon, ground, camera, telemetry)
        head = world_to_pixel(lat, lon, ground + 1.7, camera, telemetry)  # a standing person
        half_width = abs(feet[1] - head[1]) * 0.35
        truth.append(Target(
            box=(feet[0] - half_width, head[1], feet[0] + half_width, feet[1]),
            geo=(lat, lon),
            range_m=slant_range_m(telemetry, lat, lon, ground),
        ))

    # Opt in to a live server with REDIS_URL; otherwise the connection is mocked
    # so the demo never writes to someone's Redis by surprise.
    url = os.environ.get("REDIS_URL")
    if url:
        bus, backend = RedisBus.from_url(url), url
        bus.client.ping()
    else:
        bus, backend = RedisBus(FakeRedisStreams()), "mocked connection (set REDIS_URL for a server)"

    agent = DetectionAgent(bus, CASE, camera, dem=dem, tracker=BoTSORT(min_hits=3))
    rgb, lidar = yolo11m_stub(seed=11, recall=0.95), lidar_stub(seed=11, recall=0.8)

    print(f"\n  redis: {backend}")
    print(f"  {dem}\n  terrain at drone: {dem.elevation(*DRONE):.1f} m, "
          f"drone at {telemetry.altitude_m:.0f} m\n")
    for i in range(6):
        frame = f"frame_{i:04d}"
        emitted = agent.process_frame(
            frame,
            [rgb.detect(frame, truth, CASE), lidar.detect(frame, truth, CASE)],
            telemetry,
        )
        print(f"  {frame}: published {len(emitted)}")
        for clue in emitted:
            meta, spatial = clue.agent_metadata, clue.spatial_context
            if spatial.latitude is None:
                print(f"      track {meta['track_id']}  NO FIX  ({meta['geolocation']})")
                continue
            error = min(ground_distance_m((spatial.latitude, spatial.longitude), t.geo)
                        for t in truth)
            print(f"      track {meta['track_id']}  {spatial.latitude:.6f}, {spatial.longitude:.6f}"
                  f"  elev {spatial.altitude_m:7.1f} m  {meta['range_source']:<16}"
                  f" sigma {meta['range_sigma_m']:>6} m   {error:5.1f} m from truth")

    stream = agent.stream
    print(f"\n  bus: {bus}")
    print(f"  {bus.length(stream)} clues on {stream}, {agent.without_fix} without a fix")
    entries = bus.read(stream)
    print(f"  replay from id {entries[len(entries) // 2][0]}: "
          f"{len(bus.read(stream, last_id=entries[len(entries) // 2][0]))} newer clues")
    print(f"\n  last clue: {entries[-1][1].finding_summary}")
    print(f"  lineage:   {len(entries[-1][1].parent_clue_ids)} parent clues\n")
