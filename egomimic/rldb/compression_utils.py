# Description: Utility functions for compressing and decompressing video data.
import contextlib
import io
import json
import os

import av
import numpy as np
import simplejpeg


# context manager to suppress AV library output
@contextlib.contextmanager
def suppress_av_output():
    # Save the current stderr
    old_stderr = os.dup(2)
    # Open /dev/null
    null_fd = os.open(os.devnull, os.O_RDWR)
    # Redirect stderr to /dev/null
    os.dup2(null_fd, 2)
    try:
        yield
    finally:
        # Restore stderr
        os.dup2(old_stderr, 2)
        # Close file descriptors
        os.close(null_fd)
        os.close(old_stderr)


def encode_video(frames, method="JPEG", **kwargs):
    """
    Encode a sequence of images into a compressed byte string.

    :param frames: numpy array of shape [seq_len, h, w, 3]
    :param method: compression method ('JPEG' or 'H265')
    :param kwargs: additional arguments for the chosen method
    :return: (compressed_data, metadata)
    """
    seq_len, h, w, c = frames.shape
    assert c == 3, "Input frames must have 3 color channels"
    if method == "JPEG":
        quality = kwargs.get("jpeg_quality", 85)
        compressed_frames = []
        for frame in frames:
            compressed_frame = simplejpeg.encode_jpeg(
                frame, quality=quality, colorspace="RGB"
            )
            compressed_frames.append(compressed_frame)

        lengths = [len(frame) for frame in compressed_frames]
        compressed_data = b"".join(compressed_frames)
        metadata = {
            "method": "JPEG",
            "shape": frames.shape,
            "lengths": lengths,
            "quality": quality,
        }
    elif method == "H265":
        fps = kwargs.get("h265_fps", 30)
        crf = kwargs.get(
            "h265_crf", 23
        )  # Constant Rate Factor (0-51, lower is better quality)

        output = io.BytesIO()
        with suppress_av_output():
            container = av.open(output, mode="w", format="mp4")
            stream = container.add_stream("libx265", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = (
                "yuv420p"  # yuv420p is the only supported pixel format for H.265
            )
            stream.options = {"crf": str(crf)}

            for frame in frames:
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                packet = stream.encode(frame)
                container.mux(packet)

            # Flush the stream
            packet = stream.encode(None)
            container.mux(packet)
            container.close()

        compressed_data = output.getvalue()
        metadata = {"method": "H265", "shape": frames.shape, "fps": fps, "crf": crf}
    else:
        raise ValueError(f"Unsupported compression method: {method}")

    return compressed_data, json.dumps(metadata)


def decode_video(compressed_data, metadata):
    """
    Decode a compressed byte string back into a sequence of images.

    :param compressed_data: byte string of compressed video data
    :param metadata: JSON string containing compression metadata
    :return: numpy array of shape [seq_len, h, w, 3]
    """
    metadata = json.loads(metadata)
    method = metadata["method"]

    if method == "JPEG":
        lengths = metadata["lengths"]

        frames = []
        offset = 0
        for length in lengths:
            compressed_frame = compressed_data[offset : offset + length]
            frame = simplejpeg.decode_jpeg(compressed_frame, colorspace="RGB")
            frames.append(frame)
            offset += length

        return np.stack(frames)

    elif method == "H265":
        container = av.open(io.BytesIO(compressed_data))
        stream = container.streams.video[0]

        frames = []
        for frame in container.decode(stream):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)

        return np.stack(frames)

    else:
        raise ValueError(f"Unsupported compression method: {method}")
