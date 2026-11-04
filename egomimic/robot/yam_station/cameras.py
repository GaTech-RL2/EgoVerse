"""Shared camera setup and live front/wrist views using existing camera drivers."""

import time

import numpy as np


def validate_camera_devices(config, available_realsense=None):
    """Fail on missing/duplicate RealSense serials before robot initialization."""
    configured = {
        name: str(spec["serial_number"])
        for name, spec in config.items()
        if spec.get("enabled", True) and spec["type"] in ("realsense", "d405")
    }
    serials = list(configured.values())
    if len(serials) != len(set(serials)):
        raise ValueError("Configured RealSense cameras must use distinct serials")
    if not serials:
        return ()
    if available_realsense is None:
        from egomimic.robot.eva.eva_ws.src.eva.stream_d405 import (
            list_connected_serials,
        )

        available_realsense = list_connected_serials()
    available = {str(serial) for serial in available_realsense}
    missing = [
        f"{name}={serial}"
        for name, serial in configured.items()
        if serial not in available
    ]
    if missing:
        raise RuntimeError(
            "Configured RealSense cameras are unavailable: "
            + ", ".join(missing)
            + "; available: "
            + ", ".join(sorted(available))
        )
    return tuple(serials)


class CameraStream:
    """Reject disconnected streams while preserving existing driver color/layout."""

    def __init__(self, recorder, max_age):
        self.recorder, self.max_age = recorder, float(max_age)
        if not np.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError("Camera max_age must be positive")

    def get_image(self):
        stamp = self.recorder.last_frame_time
        if stamp is None or time.monotonic() - stamp > self.max_age:
            return None
        return self.recorder.get_image()

    def stop(self):
        self.recorder.stop()


def close_cameras(recorders):
    errors = []
    for recorder in recorders.values():
        try:
            recorder.stop()
        except Exception as error:
            errors.append(error)
    if errors:
        raise ExceptionGroup("Camera cleanup failed", errors)


def open_cameras(config):
    recorders, resolutions = {}, {}
    try:
        for name, spec in config.items():
            if not spec.get("enabled", True):
                continue
            height, width = int(spec["height"]), int(spec["width"])
            if spec["type"] in ("realsense", "d405"):
                from egomimic.robot.eva.eva_ws.src.eva.stream_d405 import (
                    RealSenseRecorder,
                )

                recorder = RealSenseRecorder(
                    str(spec["serial_number"]),
                    width=width,
                    height=height,
                    fps=int(spec.get("fps", 30)),
                )
            elif spec["type"] == "aria":
                from egomimic.robot.eva.eva_ws.src.eva.stream_aria import AriaRecorder

                recorder = AriaRecorder(
                    profile_name=spec.get("profile", "profile15"),
                    use_security=True,
                    height=height,
                    width=width,
                )
                recorders[name] = recorder
                recorder.start()
            else:
                raise ValueError(f"Unknown camera type: {spec['type']}")
            recorders[name] = recorder
            recorders[name] = CameraStream(recorder, spec.get("max_age", 1.0))
            resolutions[name] = (height, width)
    except BaseException:
        close_cameras(recorders)
        raise
    return recorders, resolutions


class CameraView:
    """One named window per configured camera, including live front and wrists."""

    def __init__(self, cameras, enabled=True, width=640):
        self.cameras, self.enabled, self.width = tuple(cameras), enabled, int(width)
        if self.width <= 0:
            raise ValueError("Preview width must be positive")

    def update(self, obs, recording=False):
        if not self.enabled:
            return None
        import cv2

        for name in self.cameras:
            frame = obs.get(name)
            if frame is None:
                frame = np.zeros((240, self.width, 3), dtype=np.uint8)
                status = f"{name}: waiting for camera"
            else:
                frame = cv2.resize(
                    frame,
                    (self.width, round(frame.shape[0] * self.width / frame.shape[1])),
                )
                status = f"{name} | {'RECORDING' if recording else 'LIVE'}"
            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255) if recording else (0, 255, 0),
                2,
            )
            cv2.imshow(name, frame)
        key = cv2.waitKey(1) & 0xFF
        return chr(key) if key != 255 else None

    def close(self):
        if self.enabled:
            import cv2

            cv2.destroyAllWindows()
