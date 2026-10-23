"""Synthetic Lance steps rows for unit tests."""

import numpy as np
import pyarrow as pa

from egomimic.rldb.lance.schema import HUMAN_STEPS


def vec(x, n):
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(x, np.float32).reshape(-1)), n
    )


def steps_table(rid, n, offset):
    t = np.arange(n, dtype=np.float32)
    hand = lambda s: pa.StructArray.from_arrays(  # noqa: E731
        [
            vec(np.c_[t + offset + s, np.zeros((n, 6))], 7),
            vec(np.c_[t + offset + s, np.zeros((n, 6))], 7),
            vec(np.zeros((n, 63)), 63),
            vec(np.zeros((n, 63)), 63),
            pa.array(np.ones(n, np.float32)),
        ],
        fields=list(HUMAN_STEPS.field("left").type),
    )
    valid = np.ones(n, bool)
    valid[4] = False
    return pa.table(
        {
            "recording_id": pa.array([rid] * n),
            "timestamp_ns": pa.array(np.arange(n, dtype=np.int64) * 33_333_333),
            "head_pose": vec(np.c_[t + offset, np.zeros((n, 6))], 7),
            "left": hand(0.5),
            "right": hand(0.25),
            "left_valid": pa.array(valid),
            "right_valid": pa.array(np.ones(n, bool)),
            "head_pose_valid": pa.array(np.ones(n, bool)),
            "eye_gaze": vec(np.zeros((n, 3)), 3),
        },
        schema=HUMAN_STEPS,
    )
