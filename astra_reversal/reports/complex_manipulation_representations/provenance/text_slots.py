"""Instruction-only positions in the released RoboCasa pi0.5 tokenizer.

Unlike the earlier LIBERO plain-text profile, this format includes discrete
robot state. Never interpolate Task/State/Action delimiters or state tokens.
"""

import numpy as np


def instruction_slots(tokenizer, prompt, state, token_ids, token_mask):
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    if not cleaned or state.shape != (32,) or not np.isfinite(state).all():
        raise ValueError("Expected an instruction and normalized 32D state")
    discrete = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    text = f"Task: {cleaned}, State: {' '.join(map(str, discrete))};\nAction: "
    # The proto API does not implement add_bos. Add the same BOS ID explicitly.
    proto = tokenizer.encode(text, return_type="proto")
    ids = np.asarray([tokenizer.bos_id(), *[piece.id for piece in proto.pieces]])
    actual, valid = np.asarray(token_ids), np.asarray(token_mask)
    if actual.shape != (200,) or valid.shape != (200,) or valid.dtype != np.bool_:
        raise ValueError("Expected the released 200-slot text profile")
    if len(ids) > 200:
        raise ValueError("Intervention source would truncate the native text/state")
    if not np.array_equal(
        actual, np.pad(ids, (0, 200 - len(ids)))
    ) or not np.array_equal(valid, np.arange(200) < len(ids)):
        raise ValueError("Token IDs differ from the released tokenizer format")
    selected = np.zeros(200, dtype=bool)
    encoded = text.encode("utf-8")
    for i, piece in enumerate(proto.pieces, start=1):
        # SentencePiece reports UTF-8 byte offsets, including leading whitespace.
        # Verify offsets against the literal surface before using the interval.
        if encoded[piece.begin : piece.end] != piece.surface.encode("utf-8"):
            raise ValueError("Tokenizer surface offsets cannot be verified")
        selected[i] = (
            piece.begin >= len("Task:")
            and piece.end > len("Task: ")
            and piece.end <= len("Task: ") + len(cleaned.encode("utf-8"))
        )
    if not selected.any() or np.any(selected & ~valid):
        raise ValueError("No verified instruction-only positions")
    return selected


def alignment_indices(source_mask, target_mask):
    """Left-align valid instruction slots; pad with zeros or truncate explicitly."""
    source, target = np.flatnonzero(source_mask), np.flatnonzero(target_mask)
    if not len(source) or not len(target):
        raise ValueError("Instruction masks must be nonempty")
    indices = np.zeros(len(target_mask), dtype=np.int32)
    valid = np.zeros(len(target_mask), dtype=bool)
    count = min(len(source), len(target))
    indices[target[:count]] = source[:count]
    valid[target[:count]] = True
    return indices, valid
