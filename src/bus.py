"""The event bus — Redis Streams, as Phase 0 requires.

Producers publish clues to `clues:<case_id>`; consumers resume from the last
entry id they saw. One stream per case, so a consumer reading one search never
sees another's — the same scoping `case_id` gives the evaluator.

Clues cross the wire as JSON in a single stream field, so a clue carrying
something unserializable fails at publish, in a test, rather than in flight.

Testing without a server
------------------------
`FakeRedisStreams` implements the handful of stream commands `RedisBus` uses.
It stands in for the *connection*, not for the bus, so every test still runs the
real `RedisBus` code: entry-id parsing, field encoding, XREAD resume semantics,
and byte/str decoding. Mocking one layer up would test nothing but the mock.
"""

import fnmatch
import time

from .contracts.clue import ClueContract

CLUE_FIELD = "clue"


def stream_for(case_id):
    """The stream a case's clues belong on."""
    return f"clues:{case_id}"


class RedisBus:
    """Clue transport over Redis Streams.

    `maxlen` caps a stream so a long flight cannot grow it without bound; Redis
    trims approximately, which is cheaper and is what production wants.
    """

    def __init__(self, client, maxlen=None, key_pattern="clues:*"):
        self.client = client
        self.maxlen = maxlen
        self.key_pattern = key_pattern

    @classmethod
    def from_url(cls, url="redis://localhost:6379/0", **kwargs):
        """Connect to a real server. Decoded responses keep ids and JSON as str."""
        import redis

        return cls(redis.Redis.from_url(url, decode_responses=True), **kwargs)

    def publish(self, clue, stream=None):
        """XADD a clue. Returns its entry id.

        The clue's own `case_id` picks the stream unless one is given, so a
        producer cannot accidentally publish onto another case's stream.
        """
        if not isinstance(clue, ClueContract):
            raise TypeError(f"only ClueContract may be published, got {type(clue).__name__}")

        stream = stream or stream_for(clue.case_id)
        extra = {"maxlen": self.maxlen, "approximate": True} if self.maxlen else {}
        entry_id = self.client.xadd(stream, {CLUE_FIELD: clue.model_dump_json()}, **extra)
        return _text(entry_id)

    def read(self, stream, last_id="0-0", count=None):
        """Entries after `last_id`, oldest first, as (entry_id, ClueContract).

        XREAD is the resume primitive: it returns strictly newer entries, which
        is what a consumer holding its last-seen id actually wants.
        """
        response = self.client.xread({stream: last_id}, count=count) or []
        entries = []
        for _, items in response:
            for entry_id, fields in items:
                entries.append((_text(entry_id), ClueContract.model_validate_json(_field(fields))))
        return entries

    def length(self, stream):
        return self.client.xlen(stream)

    def streams(self):
        return sorted(_text(k) for k in self.client.scan_iter(match=self.key_pattern))

    def __repr__(self):
        return f"RedisBus({', '.join(f'{s}={self.length(s)}' for s in self.streams())})"


class FakeRedisStreams:
    """In-memory stand-in for the redis-py client's stream commands.

    Mirrors the real client closely enough to be worth trusting: Redis-shaped
    monotonic `<ms>-<seq>` ids, XREAD's strictly-after semantics, approximate
    MAXLEN trimming, and bytes responses unless `decode_responses` is set — the
    real client's default is bytes, and code that only ever sees str will break
    against a live server.
    """

    def __init__(self, decode_responses=True):
        self.decode_responses = decode_responses
        self.streams = {}
        self._last_ms = 0
        self._seq = 0

    def _encode(self, value):
        return value if self.decode_responses else value.encode()

    def _next_id(self):
        now = int(time.time() * 1000)
        if now > self._last_ms:
            self._last_ms, self._seq = now, 0
        else:
            self._seq += 1
        return f"{self._last_ms}-{self._seq}"

    def xadd(self, name, fields, id="*", maxlen=None, approximate=True):
        if id != "*":
            raise NotImplementedError("only auto-generated ids are used")
        entry_id = self._next_id()
        entries = self.streams.setdefault(_text(name), [])
        entries.append((entry_id, {_text(k): _text(v) for k, v in fields.items()}))
        if maxlen is not None and len(entries) > maxlen:
            del entries[: len(entries) - maxlen]
        return self._encode(entry_id)

    def xread(self, streams, count=None, block=None):
        response = []
        for name, last_id in streams.items():
            name = _text(name)
            after = _parse_id(last_id)
            newer = [(i, f) for i, f in self.streams.get(name, ()) if _parse_id(i) > after]
            if not newer:
                continue
            if count:
                newer = newer[:count]
            response.append((
                self._encode(name),
                [(self._encode(i), {self._encode(k): self._encode(v) for k, v in f.items()})
                 for i, f in newer],
            ))
        return response

    def xlen(self, name):
        return len(self.streams.get(_text(name), ()))

    def scan_iter(self, match=None):
        for name in self.streams:
            if match is None or fnmatch.fnmatch(name, match):
                yield self._encode(name)

    def ping(self):
        return True


def _text(value):
    return value.decode() if isinstance(value, (bytes, bytearray)) else value


def _field(fields):
    """The clue JSON out of a stream entry, whether keys are bytes or str."""
    for key, value in fields.items():
        if _text(key) == CLUE_FIELD:
            return _text(value)
    raise ValueError(f"stream entry has no '{CLUE_FIELD}' field: {list(fields)}")


def _parse_id(entry_id):
    major, _, minor = _text(entry_id).partition("-")
    return (int(major), int(minor or 0))
