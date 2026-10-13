"""Request pacing changes waiting, never model payloads, retry counts or actions."""

from types import SimpleNamespace

import pytest

from astra_reversal.hardware_interface.provider import ModelFailure
from experiments.libero_hardware_interface.tools.request_admission import (
    Admission,
    retry_seconds,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_slots_are_disjoint_and_one_call_per_cycle():
    seen = []
    for index in range(8):
        clock = Clock()
        admission = Admission(index, clock=clock)
        admission.wait()
        seen.append(clock.now)
        admission.wait()
        assert clock.now == index + 8
    assert seen == list(range(8))


def test_rejected_attempt_is_not_reissued_inside_admission():
    clock = Clock()
    admission = Admission(0, clock=clock)
    events, calls = [], []
    session = SimpleNamespace(
        post=None,
        role="actor",
        events=SimpleNamespace(emit=lambda kind, **value: events.append((kind, value))),
    )
    body = {"input": ["exact same input"]}

    def original(session, path, received, timeout):
        calls.append(received)
        if len(calls) == 1:
            raise ModelFailure("provider_http_429")
        return {"ok": True}

    with pytest.raises(ModelFailure):
        admission.invoke(original, session, "responses", body, 180)
    assert len(calls) == 1
    assert admission.invoke(original, session, "responses", body, 180) == {"ok": True}
    assert clock.now >= 60
    assert all(call is body for call in calls)
    assert any(kind == "provider_rate_cooldown" for kind, _ in events)


def test_retry_after_takes_precedence_and_other_errors_propagate():
    clock = Clock()
    admission = Admission(0, clock=clock)
    admission.retry_after = 200
    assert admission.rejected() == 200
    assert retry_seconds("Thu, 01 Jan 1970 00:03:00 GMT", 0) == 180
    assert retry_seconds("123", 0) == 123
    assert retry_seconds("NaN", 0) == 0
    assert retry_seconds("invalid", 0) == 0
