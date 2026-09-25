"""Reconstruct the verified Linux Pillow blend, retaining exact image hashes.

Pillow 12.3.0 Blend.c uses float alpha and casts the interpolated result to uint8.
The experiment's x86_64 wheel performs separate float32 multiply/add; the local
arm64 wheel can fuse them. The different rounding can cross an integer boundary.
This CPU postprocessor reproduces the measured Linux arithmetic explicitly. It
does not modify the experiment's frozen apply_vision function or recorded pixels.
"""

import copy

import numpy as np
from frozen_intervention.interventions import apply_vision as frozen_apply_vision
from PIL import Image, ImageDraw

BACKEND = "Pillow-12.3.0-Linux-x86_64-separate-float32-blend"


def blend_linux(image, layer, gain):
    original = np.asarray(image, dtype=np.float32)
    changed = np.asarray(layer, dtype=np.float32)
    # Separate NumPy operations prevent a compiler from contracting the multiply
    # and add into one instruction. Both intermediate values are float32.
    scaled = np.multiply(changed - original, np.float32(gain), dtype=np.float32)
    result = np.add(original, scaled, dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vision_linux(observation, annotations):
    # Retain the frozen parser's image, camera, gain and geometry validation. Its
    # local rendered output is intentionally unused when there are annotations.
    validated = frozen_apply_vision(observation, annotations)
    if not annotations:
        return validated
    result = copy.deepcopy(observation)
    for annotation in annotations:
        gain = annotation["gain"]
        if gain == 0:
            continue
        camera = annotation["camera"]
        image = Image.fromarray(result[camera])
        layer = image.copy()
        draw = ImageDraw.Draw(layer)
        coordinates = np.asarray(annotation["coordinates"], dtype=np.float64).tolist()
        if annotation["kind"] == "box":
            draw.rectangle(coordinates, outline=(255, 0, 255), width=3)
        else:
            x, y = coordinates
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 255))
        result[camera] = blend_linux(image, layer, gain)
    return result
