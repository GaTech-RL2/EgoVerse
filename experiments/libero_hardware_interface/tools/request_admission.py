"""Space shared inference requests and honor rate-limit cooldowns without reruns.

The model's existing retry count is unchanged. Waiting occurs before a transport
attempt; a successful response is never requested or dispatched a second time.
"""

import math
import time
import urllib.error
from datetime import timezone
from email.utils import parsedate_to_datetime

from astra_reversal.hardware_interface.common import digest
from astra_reversal.hardware_interface.provider import ModelFailure, Session


def retry_seconds(value, now):
    if not value:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        try:
            stamp = parsedate_to_datetime(value)
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            result = stamp.timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, result) if math.isfinite(result) else 0.0


class Headers:
    def __init__(self, delegate, pacer):
        self.delegate, self.pacer = delegate, pacer

    def open(self, *args, **kwargs):
        try:
            return self.delegate.open(*args, **kwargs)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                self.pacer.retry_after = retry_seconds(
                    error.headers.get("Retry-After"), self.pacer.clock.time()
                )
            raise


class Admission:
    def __init__(self, index, participants=8, slot_seconds=1.0, *, clock=time):
        if not 0 <= index < participants or slot_seconds <= 0:
            raise ValueError("invalid_request_admission_assignment")
        self.index, self.participants, self.slot_seconds = (
            index,
            participants,
            slot_seconds,
        )
        self.clock = clock
        self.last_slot = -1
        self.cooldown_until = 0.0
        self.retry_after = 0.0
        self.rejections = 0

    def wait(self):
        start = self.clock.monotonic()
        while True:
            cooldown = self.cooldown_until - self.clock.monotonic()
            if cooldown > 0:
                self.clock.sleep(cooldown)
            now = self.clock.time()
            slot = max(math.floor(now / self.slot_seconds), self.last_slot + 1)
            slot += (self.index - slot) % self.participants
            delay = slot * self.slot_seconds - now
            if delay > 0:
                self.clock.sleep(delay)
                continue
            self.last_slot = slot
            return self.clock.monotonic() - start

    def rejected(self):
        self.rejections += 1
        delay = max(
            self.retry_after, min(300.0, 60.0 * 2 ** min(self.rejections - 1, 3))
        )
        self.cooldown_until = self.clock.monotonic() + delay
        self.retry_after = 0.0
        return delay

    def invoke(self, original, session, path, body, timeout):
        if hasattr(session.post, "opener") and not isinstance(
            session.post.opener, Headers
        ):
            session.post.opener = Headers(session.post.opener, self)
        waited = self.wait()
        usage = session.meter.usage[session.role]
        usage["client_admission_seconds"] = (
            usage.get("client_admission_seconds", 0.0) + waited
        )
        session.events.emit(
            "provider_admission",
            role=session.role,
            wait_seconds=waited,
            slot_index=self.index,
            participants=self.participants,
            slot_seconds=self.slot_seconds,
            request_sha256=digest(body),
        )
        try:
            result = original(session, path, body, timeout)
        except ModelFailure as error:
            if str(error) == "provider_http_429":
                delay = self.rejected()
                session.events.emit(
                    "provider_rate_cooldown",
                    role=session.role,
                    wait_before_next_attempt_seconds=delay,
                    request_sha256=digest(body),
                    retry_count_unchanged=True,
                )
            raise
        self.rejections = 0
        return result


def install(index, participants=8, slot_seconds=1.0):
    if getattr(Session._post, "admission_installed", False):
        raise ValueError("request_admission_already_installed")
    original = Session._post
    pacer = Admission(index, participants, slot_seconds)

    def admitted(session, path, body, timeout):
        return pacer.invoke(original, session, path, body, timeout)

    admitted.admission_installed = True
    Session._post = admitted
    return {
        "index": index,
        "participants": participants,
        "slot_seconds": slot_seconds,
        "minimum_429_cooldown_seconds": 60,
        "honor_retry_after": True,
        "model_retry_count_unchanged": True,
        "model_request_body_unchanged": True,
    }
