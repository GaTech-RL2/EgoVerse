"""Public sensor blocks shared by the native control transports."""
import json

from efe_pear_context import interface_context
from efe_pear_observation_check import validate_frames


def public_blocks(visible, images):
    cameras = visible['action_contract']['cameras']
    validate_frames(images, cameras)
    guide = interface_context(visible, image_sizes=[list(image.size) for image in images])
    text = guide['text'] + '\n\nPublic observation JSON:\n' + json.dumps(visible, allow_nan=False)
    blocks = [{'type': 'text', 'text': text}]
    for name, image in zip(cameras, images, strict=True):
        blocks.extend([{'type': 'text', 'text': 'Current camera: ' + name},
                       {'type': 'image', 'image': image}])
    return blocks, guide['sha256']
