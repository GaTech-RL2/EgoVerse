"""Deterministic image annotations that expire when their source frame changes."""

import copy

from .agent import CAMERAS
from .records import digest


def augment(observation, proposal, source_observation, *, visuals=True):
    result = copy.deepcopy(observation)
    events = []
    if not visuals or proposal is None:
        return result, events
    from PIL import Image, ImageDraw

    for camera in CAMERAS:
        annotations = [x for x in proposal.annotations if x["camera"] == camera]
        if not annotations:
            continue
        # No tracker is assumed. Even a one-step-old annotation is expired if its
        # camera pixels changed. Unmodified copies continue to be sent to Astra.
        if digest(observation[camera]) != digest(source_observation[camera]):
            events.append(
                {
                    "camera": camera,
                    "event": "annotation_expired",
                    "count": len(annotations),
                }
            )
            continue
        import numpy as np

        image = Image.fromarray(observation[camera])
        draw = ImageDraw.Draw(image)
        for item in annotations:
            coordinates = tuple(round(x) for x in item["coordinates"])
            if item["kind"] == "box":
                draw.rectangle(coordinates, outline=(255, 0, 255), width=2)
            else:
                x, y = coordinates
                draw.line((x - 3, y, x + 3, y), fill=(255, 0, 255), width=2)
                draw.line((x, y - 3, x, y + 3), fill=(255, 0, 255), width=2)
        result[camera] = np.asarray(image).copy()
        events.append(
            {
                "camera": camera,
                "event": "annotation_applied",
                "count": len(annotations),
                "renderer": "magenta-outline-v1",
            }
        )
    return result, events
