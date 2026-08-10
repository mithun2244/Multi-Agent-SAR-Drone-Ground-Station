"""The History agent — what happened in cases like this one before.

Architecture, Phase 5: "History — Qdrant RAG with BGE-M3 embeddings, a reranker,
and hybrid dense plus keyword search; bring the provenance allow-list guard
online with it."

None of that infrastructure exists yet, so retrieval here is a local archive
scored with TF-IDF cosine — real ranking maths over real text, just without a
vector database or a learned embedding. The seam is `Archive.search`, so
swapping in Qdrant plus BGE-M3 is a new Archive, not a change to the agent.

ponytail: TF-IDF over a handful of records. It genuinely ranks, and it will not
match a dense retriever on paraphrase — "went downhill" and "descended the
gully" share no terms. Upgrade when the archive outgrows keyword overlap.

The provenance allow-list
------------------------
A retrieval agent is the obvious way to poison this system: put a fabricated
case in the archive and the LLM will cite it. So retrieval is filtered to
sources on an allow-list *before* anything reaches the model, and blocked
records are counted. Phase 7 hardens this into the real guard; this is the
minimum that makes the allow-list mean something today.
"""

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from ..contracts.clue import AgentSource, ClueContract, SpatialContext
from ..guardrails.parsers import parse_text
from ..guardrails.provenance import TAG_HISTORY
from ..guardrails.schemas import TextReply
from .llm import DEFAULT_LLM_MODEL, LLMUnavailable

HISTORY_PROVENANCE = TAG_HISTORY
HISTORY_MODEL_TAG = DEFAULT_LLM_MODEL

# Only records from these sources may be retrieved. Anything else is dropped.
DEFAULT_ALLOWED_PROVENANCE = frozenset({"archive:regional-sar", "archive:national-sar"})

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ArchiveCase:
    """One historical search, as the archive holds it."""

    case_id: str
    summary: str
    terrain: str = ""
    subject_type: str = ""
    season: str = ""
    outcome: str = ""
    found_distance_m: int | None = None
    provenance: str = "archive:regional-sar"

    @property
    def text(self):
        return " ".join(filter(None, [
            self.summary, self.terrain, self.subject_type, self.season, self.outcome,
        ]))


# A small mock corpus standing in for the real archive.
DEFAULT_ARCHIVE = (
    ArchiveCase("2019-041", "Hiker lost in low cloud on a north-facing ridge descended a "
                "drainage and was found in a gully below the treeline.",
                terrain="alpine ridge scree gully", subject_type="adult hiker",
                season="autumn", outcome="found alive", found_distance_m=1400),
    ArchiveCase("2020-112", "Elderly walker became disoriented near a ridge path in rain and "
                "sheltered under a rock band without moving far.",
                terrain="alpine ridge boulder", subject_type="elderly walker",
                season="autumn", outcome="found alive", found_distance_m=350),
    ArchiveCase("2021-007", "Two hikers benighted on scree above a meltwater stream; both "
                "followed the watercourse downhill overnight.",
                terrain="scree stream meltwater", subject_type="adult hikers",
                season="summer", outcome="found alive", found_distance_m=2100),
    ArchiveCase("2018-233", "Ski tourer caught by weather on an exposed north ridge, found "
                "close to the last known point in a snow hole.",
                terrain="ridge snow exposed", subject_type="ski tourer",
                season="winter", outcome="found deceased", found_distance_m=200),
    ArchiveCase("2022-064", "Child separated from a family group on a valley trail was found "
                "downhill within a short distance, close to running water.",
                terrain="valley trail stream", subject_type="child",
                season="summer", outcome="found alive", found_distance_m=600),
)


def _tokens(text):
    return _TOKEN.findall(text.lower())


class Archive:
    """Local case archive with TF-IDF cosine retrieval."""

    def __init__(self, cases=DEFAULT_ARCHIVE, allowed_provenance=DEFAULT_ALLOWED_PROVENANCE):
        self.allowed_provenance = frozenset(allowed_provenance)
        self.blocked = 0

        allowed = []
        for case in cases:
            if case.provenance in self.allowed_provenance:
                allowed.append(case)
            else:
                self.blocked += 1
        self.cases = tuple(allowed)

        # IDF over the admitted corpus only: a blocked record must not even
        # influence how the rest are scored.
        documents = [set(_tokens(c.text)) for c in self.cases]
        total = max(1, len(documents))
        self.idf = {
            term: math.log((total + 1) / (1 + sum(term in d for d in documents))) + 1.0
            for term in set().union(*documents) if documents
        }
        self.vectors = [self._vector(c.text) for c in self.cases]

    def _vector(self, text):
        counts = Counter(_tokens(text))
        if not counts:
            return {}
        longest = max(counts.values())
        return {t: (n / longest) * self.idf.get(t, 1.0) for t, n in counts.items()}

    def search(self, query, k=3, min_score=0.02):
        """Best-matching cases, highest score first."""
        query_vector = self._vector(query)
        if not query_vector:
            return []

        scored = []
        for case, vector in zip(self.cases, self.vectors):
            score = _cosine(query_vector, vector)
            if score >= min_score:
                scored.append((round(score, 4), case))
        scored.sort(key=lambda pair: (-pair[0], pair[1].case_id))
        return scored[:k]


def _cosine(a, b):
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[t] * b[t] for t in shared)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def synthesise(hits, query, complete=None):
    """Turn retrieved cases into one insight. Returns (text, source).

    The model summarises retrieved records and nothing else — the prompt hands
    it the cases and forbids adding any. Losing the model loses the prose, not
    the retrieval.
    """
    if not hits:
        return ("No comparable historical cases found.", "computed-fallback")

    if complete is not None:
        try:
            reply = complete(_prompt(hits, query))
        except LLMUnavailable:
            reply = None
        if reply is not None:
            result = parse_text(reply, TextReply)
            if result.ok:
                return (result.value.text, HISTORY_MODEL_TAG)

    distances = [c.found_distance_m for _, c in hits if c.found_distance_m is not None]
    parts = [f"{len(hits)} comparable case(s) retrieved."]
    if distances:
        parts.append(
            f"Subjects were found between {min(distances)} m and {max(distances)} m from "
            f"the last known point (median {sorted(distances)[len(distances) // 2]} m)."
        )
    parts.append("Closest match: " + hits[0][1].summary)
    return (" ".join(parts), "computed-fallback")


def _prompt(hits, query):
    lines = [
        "You are briefing a search planner on comparable past incidents.",
        f"Current search: {query}",
        "",
        "These records were RETRIEVED from the case archive. Use only these.",
        "Do not add cases, statistics, or locations that are not listed here.",
        "",
    ]
    for score, case in hits:
        distance = f"{case.found_distance_m} m from last known point" if case.found_distance_m else "distance unrecorded"
        lines.append(
            f"  [{case.case_id}] (match {score:.2f}) {case.summary} "
            f"Subject: {case.subject_type}. Season: {case.season}. "
            f"Outcome: {case.outcome}, {distance}."
        )
    lines += [
        "",
        "In at most 90 words, say what these cases suggest about where to "
        "concentrate this search. Cite case ids. No preamble.",
    ]
    return "\n".join(lines)


class HistoryAgent:
    """Retrieves comparable cases and publishes the synthesis."""

    def __init__(self, bus, case_id, archive=None, complete=None, stream=None, top_k=3):
        self.bus = bus
        self.case_id = case_id
        self.archive = archive if archive is not None else Archive()
        self.complete = complete
        self.stream = stream
        self.top_k = top_k
        self.published = 0
        self.empty_queries = 0

    def recall(self, query, position=None, at=None):
        """Search the archive and publish what it found. None if nothing matched.

        Publishing an empty insight would put "we know of nothing similar" on the
        bus as though it were a finding; silence is the honest answer.
        """
        hits = self.archive.search(query, k=self.top_k)
        if not hits:
            self.empty_queries += 1
            return None

        insight, source = synthesise(hits, query, self.complete)
        clue = self.build_clue(query, hits, insight, source, position, at)
        self.bus.publish(clue, stream=self.stream)
        self.published += 1
        return clue

    def build_clue(self, query, hits, insight, source, position=None, at=None):
        observed_at = at or datetime.now(timezone.utc)
        distances = [c.found_distance_m for _, c in hits if c.found_distance_m is not None]
        return ClueContract(
            clue_id=str(uuid.uuid5(
                uuid.NAMESPACE_URL, f"sar:history:{self.case_id}:{query}"
            )),
            case_id=self.case_id,
            timestamp=observed_at,
            source_agent=AgentSource.HISTORY_RAG,
            # Retrieval strength, not belief that anyone is anywhere. A weak
            # best match means the archive had little to say.
            confidence_score=round(min(1.0, hits[0][0] * 2.0), 4),
            finding_summary=insight[:400],
            spatial_context=SpatialContext(
                latitude=position[0] if position else None,
                longitude=position[1] if position else None,
            ),
            provenance_tag=HISTORY_PROVENANCE,
            agent_metadata={
                "insight": insight,
                "insight_source": source,
                "query": query,
                "retrieved": [
                    {"case_id": c.case_id, "score": s, "outcome": c.outcome,
                     "found_distance_m": c.found_distance_m, "provenance": c.provenance}
                    for s, c in hits
                ],
                "found_distance_m_range": [min(distances), max(distances)] if distances else None,
                "blocked_by_allow_list": self.archive.blocked,
            },
        )
