"""Semantic validation in addition to the discoverable JSON Schema."""

import math

from .common import exact


def validate_description(device):
    exact(device, ("schema_version", "device_id", "observations", "actions"))
    if device["schema_version"] != "hardware-1":
        raise ValueError("hardware_version")
    seen = set()
    for group in ("observations", "actions"):
        for channel in device[group]:
            if channel["channel"] in seen:
                raise ValueError("duplicate_channel")
            seen.add(channel["channel"])
            if any(type(v) is not int or v <= 0 for v in channel["shape"]):
                raise ValueError("channel_shape")
            if (
                type(channel["frequency_hz"]) not in (int, float)
                or not math.isfinite(channel["frequency_hz"])
                or channel["frequency_hz"] <= 0
            ):
                raise ValueError("channel_frequency")
            if group == "actions":
                if channel["control_frequency_hz"] != channel["frequency_hz"]:
                    raise ValueError("control_frequency")
                size = math.prod(channel["shape"])
                for name in ("limits", "safe_range"):
                    bounds = channel[name]
                    exact(bounds, ("min", "max"))
                    if any(
                        type(v) is not list or len(v) != size for v in bounds.values()
                    ):
                        raise ValueError("bounds_shape")
                    if any(
                        type(x) not in (int, float) or not math.isfinite(x)
                        for values in bounds.values()
                        for x in values
                    ):
                        raise ValueError("bounds_finite")
                    if any(a > b for a, b in zip(bounds["min"], bounds["max"])):
                        raise ValueError("inverted_bounds")
                if any(
                    s < a or t > b
                    for s, t, a, b in zip(
                        channel["safe_range"]["min"],
                        channel["safe_range"]["max"],
                        channel["limits"]["min"],
                        channel["limits"]["max"],
                    )
                ):
                    raise ValueError("safe_bounds_outside_limits")
    return True
