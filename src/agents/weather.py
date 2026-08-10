"""The Weather agent — the second source, and the first one that is not a sensor.

Phase 4 (docs/architecture.md): "Weather comes second, and the choice is
deliberate. It is API-only, so it brings no LLM, no untrusted input, and needs no
guard. What it does bring is a second source, which is the point: corroboration
and confidence-weighting now do real work instead of sitting idle. The
hypothermia-risk indicator and the survival window are derived at this stage."

Data comes from Open-Meteo, which is free and needs no key, over stdlib
`urllib`. The fetcher is injected, so tests and demos run offline against fixed
conditions and never depend on someone else's uptime.

On the survival model
---------------------
Wind chill is the Environment Canada / NWS formula — standard and citable. The
survival window is *not*: it is a coarse band table over apparent temperature,
and it is the weakest number this system produces. It is deliberately built as a
visible, tunable table rather than buried arithmetic, because a survival
estimate that looks authoritative and is wrong is worse than one that is
obviously a heuristic. Phase 5's Health agent refines it with the subject's own
profile; Phase 8's critic retunes it against logged outcomes.
"""

import json
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.provenance import TAG_WEATHER

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_PROVENANCE = TAG_WEATHER

# Apparent temperature at or below which exposure is a hypothermia risk. Wet
# subjects lose heat far faster, so the wet threshold is much higher.
RISK_DRY_C = 5.0
RISK_WET_C = 10.0

# ponytail: coarse band table, apparent temperature -> survivable hours. Good
# enough to drive triage ordering, not good enough to tell a family. Replace
# with a validated model once Phase 5 supplies subject age, clothing and injury.
WINDOW_BANDS = (
    (-10.0, 2),
    (-5.0, 3),
    (0.0, 6),
    (5.0, 12),
    (10.0, 24),
    (15.0, 48),
)
MAX_WINDOW_HOURS = 72


@dataclass(frozen=True)
class Conditions:
    """What the API said, normalised."""

    temperature_c: float
    wind_kmh: float = 0.0
    precipitation_mm: float | None = None
    humidity_pct: float | None = None
    observed_at: datetime | None = None


@dataclass(frozen=True)
class Assessment:
    """What it means for someone lying out in it."""

    apparent_c: float
    wet: bool
    hypothermia_risk: bool
    survival_window_hours: int


def wind_chill_c(temperature_c, wind_kmh):
    """Environment Canada / NWS wind chill index, in Celsius.

    Only defined for cold, breezy conditions; outside that domain the apparent
    temperature is just the air temperature, which is what the standard says to
    report rather than extrapolating the formula somewhere it does not hold.
    """
    if temperature_c > 10.0 or wind_kmh <= 4.8:
        return temperature_c
    v = wind_kmh ** 0.16
    return 13.12 + 0.6215 * temperature_c - 11.37 * v + 0.3965 * temperature_c * v


def assess(conditions, risk_dry_c=RISK_DRY_C, risk_wet_c=RISK_WET_C):
    """Derive hypothermia risk and a survival window from conditions."""
    apparent = wind_chill_c(conditions.temperature_c, conditions.wind_kmh)
    wet = bool(
        (conditions.precipitation_mm or 0.0) > 0.2
        or (conditions.humidity_pct is not None and conditions.humidity_pct >= 90.0)
    )

    risk = apparent <= (risk_wet_c if wet else risk_dry_c)

    hours = MAX_WINDOW_HOURS
    for ceiling, band_hours in WINDOW_BANDS:
        if apparent < ceiling:
            hours = band_hours
            break
    if wet:
        # Wet clothing strips heat several times faster than still air.
        hours = max(1, hours // 2)

    return Assessment(
        apparent_c=round(apparent, 2),
        wet=wet,
        hypothermia_risk=risk,
        survival_window_hours=int(hours),
    )


def open_meteo_fetch(latitude, longitude, timeout=10.0, url=OPEN_METEO_URL):
    """Current conditions from Open-Meteo. Free, no key, stdlib only."""
    query = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    })
    with urllib.request.urlopen(f"{url}?{query}", timeout=timeout) as response:
        payload = json.loads(response.read().decode())
    return parse_open_meteo(payload)


def parse_open_meteo(payload):
    """Normalise an Open-Meteo response into `Conditions`."""
    current = payload.get("current") or {}
    if "temperature_2m" not in current:
        raise ValueError(f"no current temperature in response: {sorted(current)}")

    observed_at = None
    if current.get("time"):
        observed_at = datetime.fromisoformat(current["time"]).replace(tzinfo=timezone.utc)

    return Conditions(
        temperature_c=float(current["temperature_2m"]),
        wind_kmh=float(current.get("wind_speed_10m") or 0.0),
        precipitation_mm=_optional_float(current.get("precipitation")),
        humidity_pct=_optional_float(current.get("relative_humidity_2m")),
        observed_at=observed_at,
    )


def static_fetch(**conditions):
    """A fetcher that always returns the same conditions. For tests and demos."""
    fixed = Conditions(**conditions)
    return lambda latitude, longitude: fixed


class WeatherAgent:
    """Publishes conditions and what they mean for survivability."""

    def __init__(self, bus, case_id, fetch=None, stream=None, endpoint=OPEN_METEO_URL):
        self.bus = bus
        self.case_id = case_id
        self.fetch = fetch or open_meteo_fetch
        self.stream = stream
        # Which endpoint the data came from, so the provenance guard can check
        # it against the authorised list rather than trusting the tag alone.
        self.endpoint = endpoint
        self.published = 0

    def observe(self, latitude, longitude):
        """Fetch, assess, publish. Returns the clue that went on the bus."""
        conditions = self.fetch(latitude, longitude)
        verdict = assess(conditions)
        clue = self.build_clue(latitude, longitude, conditions, verdict)
        self.bus.publish(clue, stream=self.stream)
        self.published += 1
        return clue

    def build_clue(self, latitude, longitude, conditions, verdict):
        observed_at = conditions.observed_at or datetime.now(timezone.utc)
        return ClueContract(
            clue_id=str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"sar:weather:{self.case_id}:{latitude:.4f}:{longitude:.4f}:{observed_at.isoformat()}",
            )),
            case_id=self.case_id,
            timestamp=observed_at,
            source_agent=AgentSource.WEATHER_API,
            confidence_score=_confidence(conditions),
            finding_summary=_summary(verdict, conditions),
            # A forecast point, not a detection: no bounding box, no frame, no
            # class. Optional contract fields exist exactly so this validates.
            spatial_context=SpatialContext(latitude=latitude, longitude=longitude),
            provenance_tag=WEATHER_PROVENANCE,
            agent_metadata={
                "hypothermia_risk": verdict.hypothermia_risk,
                "survival_window_hours": verdict.survival_window_hours,
                "temperature_c": conditions.temperature_c,
                "wind_kmh": conditions.wind_kmh,
                "apparent_c": verdict.apparent_c,
                "wet": verdict.wet,
                "precipitation_mm": conditions.precipitation_mm,
                "humidity_pct": conditions.humidity_pct,
                "window_model": "banded-apparent-temperature-v1",
                "endpoint": self.endpoint,
            },
        )


def _confidence(conditions):
    """How much of the picture the API actually gave us.

    Missing precipitation or humidity means the wet/dry call is a guess, and the
    wet/dry call is what moves the survival window most.
    """
    score = 0.9
    if conditions.precipitation_mm is None:
        score -= 0.1
    if conditions.humidity_pct is None:
        score -= 0.1
    return round(max(0.5, score), 3)


def _summary(verdict, conditions):
    state = "wet" if verdict.wet else "dry"
    risk = "hypothermia risk" if verdict.hypothermia_risk else "no hypothermia risk"
    return (
        f"{conditions.temperature_c:.1f} C air, {conditions.wind_kmh:.0f} km/h wind, {state}; "
        f"feels like {verdict.apparent_c:.1f} C — {risk}, "
        f"survival window about {verdict.survival_window_hours} h"
    )


def _optional_float(value):
    return None if value is None else float(value)
