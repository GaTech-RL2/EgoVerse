"""Require actual, labelled RGB observations before admitting a simulator."""
import hashlib
import math

from PIL import ImageStat


def validate_frames(images,cameras):
    if not images or len(images)!=len(cameras):raise ValueError('Native camera frames are missing or mislabelled')
    evidence={}
    for name,image in zip(cameras,images,strict=True):
        if image.mode!='RGB'or min(image.size)<16:raise ValueError('Native camera is not a usable RGB frame')
        deviation=ImageStat.Stat(image).stddev
        if any(not math.isfinite(v)for v in deviation):raise ValueError('Nonfinite native RGB statistics')
        evidence[name]={'width':image.width,'height':image.height,'channel_stddev':deviation,
                        'pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest()}
    if all(max(row['channel_stddev'])<1 for row in evidence.values()):
        raise ValueError('Every native camera frame is blank or constant')
    return evidence
