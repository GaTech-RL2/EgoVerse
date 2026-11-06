"""Login throttling: sliding-window counts kept in memory, checked before any password
hashing (scrypt is deliberately expensive, so unthrottled it is a DoS lever).

Limits are temporary (a window, not a lockout), so an attacker cannot lock a vendor out
for good. Three keys:
- (ip, user) failures: guessing one account from one place;
- ip attempts: one source trying many accounts, or flooding scrypt;
- user failures: a looser cap on one account guessed from many places.

In-memory, so per process: the dashboard runs as one process (DEPLOY.md), and the AWS
edge limits requests per IP in front of it as well.
"""

from __future__ import annotations

import threading
import time
from collections import deque

WINDOW = 15 * 60  # seconds
PAIR_FAILURES = 5
IP_ATTEMPTS = 30
USER_FAILURES = 50
MAX_KEYS = 100_000  # memory bound; the oldest keys go first


class Throttle:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits: dict[tuple, deque] = {}
        self._lock = threading.Lock()

    def _count(self, key: tuple, now: float) -> deque:
        hits = self._hits.get(key)
        if hits is None:
            hits = self._hits[key] = deque()
        while hits and hits[0] <= now - WINDOW:
            hits.popleft()
        return hits

    def retry_after(self, ip: str, user: str) -> int:
        """Seconds until this login may be tried, 0 = go ahead (and count it)."""
        with self._lock:
            now = self._clock()
            waits = []
            for key, limit in (
                (("pair", ip, user), PAIR_FAILURES),
                (("ip", ip), IP_ATTEMPTS),
                (("user", user), USER_FAILURES),
            ):
                hits = self._count(key, now)
                if len(hits) >= limit:
                    waits.append(hits[0] + WINDOW - now)
            if waits:
                return max(1, int(max(waits) + 0.999))
            self._count(("ip", ip), now).append(now)  # every attempt counts per ip
            self._prune()
            return 0

    def failed(self, ip: str, user: str) -> None:
        with self._lock:
            now = self._clock()
            self._count(("pair", ip, user), now).append(now)
            self._count(("user", user), now).append(now)
            self._prune()

    def succeeded(self, ip: str, user: str) -> None:
        with self._lock:
            self._hits.pop(("pair", ip, user), None)

    def _prune(self) -> None:
        if len(self._hits) > MAX_KEYS:
            now = self._clock()
            for key in [
                k for k, v in self._hits.items() if not v or v[-1] <= now - WINDOW
            ]:
                del self._hits[key]
            while len(self._hits) > MAX_KEYS:
                del self._hits[next(iter(self._hits))]
