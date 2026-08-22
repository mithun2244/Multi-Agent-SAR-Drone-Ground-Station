"""A small response cache, so repeated operator queries do not burn API quota.

The NIM free tier has rate limits, and a dispatch runs seven agents. An operator
pressing the query button three times in a minute should not cost three archive
lookups and three field briefings when nothing has changed.

Why not `functools.lru_cache`
----------------------------
It would cover the caching and give hit counts for free, and if that were all
this needed it would be the right answer. Two things it cannot do matter here:

  * **TTL.** A weather-derived briefing that never expires is a safety problem,
    not a saving. Entries age out.
  * **Invalidation.** A new sortie changes the facts, and the caller needs to be
    able to say so. `lru_cache` only offers clear-everything.

Roughly forty lines buys both. Keys hash the prompt *and* any image bytes, so
two frames sharing a prompt are never confused for one another.
"""

import hashlib
import time
from collections import OrderedDict


class ResponseCache:
    """LRU with an optional time-to-live, wrapping any completer."""

    def __init__(self, maxsize=128, ttl_seconds=None, clock=time.monotonic):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    @property
    def size(self):
        return len(self._entries)

    @property
    def calls_saved(self):
        """What this actually bought: API calls not made."""
        return self.hits

    def key(self, prompt, image=None, mime_type="image/jpeg"):
        digest = hashlib.sha256()
        digest.update((prompt or "").encode())
        # One image or several — the Scene agent sends a crop and the frame it
        # came from. Each is hashed in order, so the same two pictures swapped
        # round are a different key.
        images = [image] if isinstance(image, (bytes, bytearray)) else list(image or ())
        for each in images:
            digest.update(b"\x00")
            digest.update(mime_type.encode())
            digest.update(hashlib.sha256(each).digest())
        return digest.hexdigest()

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self.ttl_seconds is not None and self.clock() - stored_at > self.ttl_seconds:
            del self._entries[key]
            self.expirations += 1
            return None
        self._entries.move_to_end(key)
        return (value,)  # wrapped so a cached None or "" is still a hit

    def set(self, key, value):
        self._entries[key] = (self.clock(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
            self.evictions += 1
        return value

    def clear(self):
        """Drop everything. Call when the facts change under the cache."""
        self._entries.clear()

    def invalidate(self, prompt, image=None, mime_type="image/jpeg"):
        return self._entries.pop(self.key(prompt, image, mime_type), None) is not None

    def wrap(self, complete):
        """A completer that serves repeats from memory.

        Failures are never cached: an `LLMUnavailable` is a fact about the
        network at one moment, and remembering it would keep an agent degraded
        long after the model came back.
        """
        def cached(prompt, image=None, mime_type="image/jpeg"):
            key = self.key(prompt, image, mime_type)
            hit = self.get(key)
            if hit is not None:
                self.hits += 1
                return hit[0]
            self.misses += 1
            return self.set(key, complete(prompt, image=image, mime_type=mime_type))

        cached.cache = self
        return cached

    def __repr__(self):
        return (f"ResponseCache({self.size}/{self.maxsize} entries, "
                f"{self.hits} hit(s), {self.misses} miss(es))")
