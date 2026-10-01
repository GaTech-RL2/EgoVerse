"""Observe transport liveness independently of long-running provider jobs."""

import json
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from astra_reversal.codex_relay import PROTOCOL_VERSION


def healthy_relay(port, token):
    request = Request(
        f"http://127.0.0.1:{port}/health", headers={"Authorization": "Bearer " + token}
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
            value = json.loads(response.read(4096))
        return value.get("protocol") == PROTOCOL_VERSION and value.get("status") in (
            "idle",
            "pending",
        )
    except (OSError, URLError, ValueError):
        return False


@dataclass
class ForwardHealth:
    started: float
    seen_healthy: bool = False
    failures: int = 0

    def observe(self, healthy, now):
        if healthy:
            self.seen_healthy, self.failures = True, 0
        elif now - self.started >= 15:
            self.failures += 1
        if now - self.started >= 15 * 60:
            return "periodic_transport_renewal"
        if self.seen_healthy and self.failures >= 3:
            return "three_failed_health_checks"
        return None
