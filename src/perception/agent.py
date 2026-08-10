"""The detection agent — the Phase 2 vertical slice, end to end.

    sensors -> weighted box fusion -> BoT-SORT -> geolocation -> bus

Architecture (Phase 2): "the agent emits one clue per confirmed track to the
bus." Only confirmed tracks are published: a single-frame blip is not a target,
and publishing one would put a phantom on the operator's map.

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

        self.published = 0
        self.without_fix = 0
        self._last_timestamp = None

    def process_frame(self, frame_id, sensor_clues, telemetry, camera_motion=None):
        """Run one frame through the pipeline. Returns the clues published."""
        fused = weighted_box_fusion(
            sensor_clues,
            weights=self.fusion_weights,
            iou_threshold=self.fusion_iou_threshold,
        )
        for clue in fused:
            if clue.case_id != self.case_id:
                raise ValueError(
                    f"clue {clue.clue_id} belongs to {clue.case_id}, not {self.case_id}"
                )

        if fused:
            self._last_timestamp = max(c.timestamp for c in fused)

        confirmed = self.tracker.update(fused, frame_id=frame_id, camera_motion=camera_motion)

        published = []
        for track in confirmed:
            clue = self._track_clue(track, frame_id, telemetry)
            self.bus.publish(clue, stream=self.stream)
            published.append(clue)
            self.published += 1
        return published

    def _track_clue(self, track, frame_id, telemetry):
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
        metadata = {
            "track_id": track.track_id,
            "track_state": track.state.value,
            "hits": track.hits,
            "frames_seen": track.frames_seen,
            "first_frame_id": track.first_frame_id,
            "geolocation": "located" if fix else "no_fix",
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
            finding_summary=self._summary(track, label, fix),
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
    def _summary(track, label, fix):
        seen = f"track {track.track_id}, {track.frames_seen} frames"
        if fix is None:
            return f"Confirmed {label} ({seen}); position unavailable, no range to terrain"
        how = "measured LiDAR range" if fix.is_measured else "range inferred from terrain"
        return (
            f"Confirmed {label} ({seen}) at {fix.latitude:.6f}, {fix.longitude:.6f}, "
            f"{fix.range_m:.1f} m away by {how}"
        )


if __name__ == "__main__":  # a full frame-by-frame run, for eyeballing the bus
    import os

    from ..bus import FakeRedisStreams, RedisBus
    from .detectors import Target, lidar_stub, yolo11n_stub
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
    rgb, lidar = yolo11n_stub(seed=11, recall=0.95), lidar_stub(seed=11, recall=0.8)

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
