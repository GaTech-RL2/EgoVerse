"""EgoVerse's own MCAP message types (raw data spec 0.1).

Only what Foxglove's schemas do not cover. Video, camera calibration, transforms and head
pose use `foxglove.CompressedVideo`, `foxglove.CameraCalibration`, `foxglove.FrameTransforms`
and `foxglove.PoseInFrame` (package `foxglove-schemas-protobuf`). Robot vendors use the
ArmState / ArmAction / GripperState below; ABC-130k's own robot types are normalized to
them before validation.

The messages are declared once, here, and built into a descriptor pool at import time, so
no `protoc` step is needed. `render_proto()` prints the same definitions as the
`proto/egoverse.proto` file handed to vendors; a unit test keeps the two in sync.

A message's time is its MCAP log time (absolute Unix ns), so none of them carry a timestamp;
spans (Task, Annotation) add only their end time.

    python -m egomimic.rldb.mcap.schemas > egomimic/rldb/mcap/proto/egoverse.proto
"""

from dataclasses import dataclass, field

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

PACKAGE = "egoverse"
PROTO_FILE = "egoverse.proto"

_F = descriptor_pb2.FieldDescriptorProto
_SCALARS = {
    "double": _F.TYPE_DOUBLE,
    "float": _F.TYPE_FLOAT,
    "int64": _F.TYPE_INT64,
    "string": _F.TYPE_STRING,
    "int32": _F.TYPE_INT32,
    "bool": _F.TYPE_BOOL,
}


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # a scalar above, an enum of the same message, or another message here
    number: int  # wire tag: never changed or reused; a new field takes a new number
    repeated: bool = False
    doc: str = ""


@dataclass(frozen=True)
class Enum:
    name: str
    values: tuple[
        str, ...
    ]  # numbered 0, 1, ...; proto3 needs the 0 value to mean "unset"


@dataclass(frozen=True)
class Message:
    name: str
    doc: str
    fields: tuple[Field, ...]
    enums: tuple[Enum, ...] = field(default=())


MESSAGES = (
    Message(
        "Vector3",
        "A 3D vector.",
        (Field("x", "double", 1), Field("y", "double", 2), Field("z", "double", 3)),
    ),
    Message(
        "Quaternion",
        "A rotation; must be unit length.",
        (
            Field("x", "double", 1),
            Field("y", "double", 2),
            Field("z", "double", 3),
            Field("w", "double", 4),
        ),
    ),
    Message(
        "Pose",
        "A child frame's pose in its parent (T_parent_child): m and a unit quaternion.",
        (Field("position", "Vector3", 1), Field("orientation", "Quaternion", 2)),
    ),
    Message(
        "CameraInfoExtra",
        "Optional per-camera facts foxglove.CameraCalibration has no field for, on the\n"
        "camera's <topic>-info-extra, published once next to <topic>-info.",
        (
            Field("frame_id", "string", 1, doc="the camera's frame, as in its -info"),
            Field(
                "valid_radius",
                "double",
                2,
                doc="px from the principal point where the lens model holds (fisheye); "
                "0 = the whole image",
            ),
            Field(
                "upright_rotation_cw_deg",
                "int32",
                3,
                doc="clockwise turn (0, 90, 180, 270) that makes a frame upright, for "
                "sensors mounted sideways",
            ),
        ),
    ),
    Message(
        "Imu",
        "One raw IMU sample, straight off the device: no gravity removal, no fusion.",
        (
            Field(
                "frame_id", "string", 1, doc="IMU frame; must be linked in /tf_static"
            ),
            Field("linear_acceleration", "Vector3", 2, doc="m/s^2, gravity included"),
            Field("angular_velocity", "Vector3", 3, doc="rad/s"),
        ),
    ),
    Message(
        "ImuInfo",
        "IMU noise model, as SLAM/VIO tools expect it. Published once. Datasheet values\n"
        "are fine when the device does not calibrate them.",
        (
            Field("frame_id", "string", 1),
            Field("accel_noise_density", "double", 2, doc="m/s^2/sqrt(Hz)"),
            Field("accel_random_walk", "double", 3, doc="m/s^3/sqrt(Hz)"),
            Field("gyro_noise_density", "double", 4, doc="rad/s/sqrt(Hz)"),
            Field("gyro_random_walk", "double", 5, doc="rad/s^2/sqrt(Hz)"),
        ),
    ),
    Message(
        "TrackingQuality",
        "State of the vendor's SLAM at this time, one per head pose.",
        (Field("state", "State", 1),),
        enums=(Enum("State", ("UNKNOWN", "OK", "LOST", "RELOCALIZED")),),
    ),
    Message(
        "HandKeypoints",
        "One hand at one time: its joints and/or the wrist pose. Joints are in the order\n"
        "the episode's hand-joints key lists them (names from spec.HAND_JOINTS). Set\n"
        "valid = false when the hand is not tracked; never write zeros in its place.\n"
        "A device without keypoints leaves positions empty and sets wrist_pose.",
        (
            Field("frame_id", "string", 1, doc="frame positions and wrist_pose are in"),
            Field(
                "positions",
                "double",
                2,
                repeated=True,
                doc="(x, y, z) in m per hand-joints entry, flattened; or empty",
            ),
            Field(
                "confidence",
                "float",
                3,
                doc="whole-hand tracking confidence in [0, 1]",
            ),
            Field("valid", "bool", 4, doc="hand tracked and its values usable"),
            Field(
                "joint_confidence",
                "float",
                5,
                repeated=True,
                doc="optional: per-joint confidences in [0, 1], one per joint",
            ),
            Field("wrist_pose", "Pose", 6, doc="optional: T_frame_wrist"),
        ),
    ),
    Message(
        "EyeGaze",
        "Where the wearer looks, as in Aria MPS eye gaze. frame_id is centered between\n"
        "the eyes with +x to the wearer's left, +y up, +z forward; the gaze point is\n"
        "depth * normalize(tan(yaw), tan(pitch), 1). Set valid = false when gaze is not\n"
        "tracked; never write zeros in its place.",
        (
            Field("frame_id", "string", 1, doc="gaze frame; linked in /tf_static"),
            Field("yaw", "double", 2, doc="rad, positive = looking left"),
            Field("pitch", "double", 3, doc="rad, positive = looking up"),
            Field("depth", "double", 4, doc="m, distance to the gaze point"),
            Field("valid", "bool", 5, doc="gaze tracked and values usable"),
        ),
    ),
    Message(
        "ArmState",
        "Measured state of one arm. Joint arrays follow joint_names. ee_pose is\n"
        "T_base_ee: the end effector in the arm-base frame (frame_id), row-major 4x4.",
        (
            Field("frame_id", "string", 1, doc='arm-base frame, e.g. "left_arm_base"'),
            Field("joint_names", "string", 2, repeated=True),
            Field("position", "double", 3, repeated=True, doc="rad (or m, prismatic)"),
            Field("velocity", "double", 4, repeated=True, doc="rad/s (or m/s)"),
            Field("torque", "double", 5, repeated=True, doc="N*m (or N)"),
            Field("ee_pose", "double", 6, repeated=True, doc="16 values, row-major"),
        ),
    ),
    Message(
        "ArmAction",
        "Command sent to one arm at this time (the target it was told to reach),\n"
        "same conventions as ArmState.",
        (
            Field("frame_id", "string", 1, doc="arm-base frame"),
            Field("joint_names", "string", 2, repeated=True),
            Field("position", "double", 3, repeated=True, doc="rad (or m, prismatic)"),
            Field("ee_pose", "double", 4, repeated=True, doc="16 values, row-major"),
        ),
    ),
    Message(
        "GripperState",
        "Gripper opening, measured (ee-state) or commanded (ee-action).",
        (Field("position", "double", 1, doc="0 = closed, 1 = fully open"),),
    ),
    Message(
        "Task",
        "One task performed in the recording; a recording may hold several\n"
        "(e.g. iron, then fold, then pack). Starts at the message's log time.",
        (
            Field("name", "string", 1, doc='task slug, e.g. "fold_clothes"'),
            Field("description", "string", 2, doc="free text for this instance"),
            Field("objects", "string", 3, repeated=True, doc="object slugs involved"),
            Field("end_time_unix_ns", "int64", 4, doc="task end, absolute Unix ns"),
        ),
    ),
    Message(
        "Annotation",
        "A label over [start, end), start being the message's log time; it must lie\n"
        "inside one /task (or the recording, without /task). Field `data` matches\n"
        "ABC-130k's /subtask-annotation.",
        (
            Field("data", "string", 1, doc="English text"),
            Field(
                "end_time_unix_ns",
                "int64",
                2,
                doc="end, absolute Unix ns; 0 = point event (one frame wide)",
            ),
            Field("annotation_type", "string", 3, doc='default "language"'),
            Field(
                "annotation_source",
                "string",
                4,
                doc="who wrote it; default the episode's lab",
            ),
        ),
    ),
)


def file_descriptor_proto() -> descriptor_pb2.FileDescriptorProto:
    fp = descriptor_pb2.FileDescriptorProto(
        name=PROTO_FILE, package=PACKAGE, syntax="proto3"
    )
    for m in MESSAGES:
        mp = fp.message_type.add(name=m.name)
        for e in m.enums:
            ep = mp.enum_type.add(name=e.name)
            for number, value in enumerate(e.values):
                ep.value.add(name=value, number=number)
        enum_names = {e.name for e in m.enums}
        for f in m.fields:
            label = _F.LABEL_REPEATED if f.repeated else _F.LABEL_OPTIONAL
            fd = mp.field.add(name=f.name, number=f.number, label=label)
            if f.kind in _SCALARS:
                fd.type = _SCALARS[f.kind]
            elif f.kind in enum_names:
                fd.type = _F.TYPE_ENUM
                fd.type_name = f".{PACKAGE}.{m.name}.{f.kind}"
            else:
                fd.type = _F.TYPE_MESSAGE
                fd.type_name = f".{PACKAGE}.{f.kind}"
    return fp


# Own pool, so importing this module never collides with another egoverse.proto.
_pool = descriptor_pool.DescriptorPool()
_pool.Add(file_descriptor_proto())


def _message_class(name: str):
    return message_factory.GetMessageClass(
        _pool.FindMessageTypeByName(f"{PACKAGE}.{name}")
    )


Vector3 = _message_class("Vector3")
Quaternion = _message_class("Quaternion")
Pose = _message_class("Pose")
CameraInfoExtra = _message_class("CameraInfoExtra")
Imu = _message_class("Imu")
ImuInfo = _message_class("ImuInfo")
TrackingQuality = _message_class("TrackingQuality")
HandKeypoints = _message_class("HandKeypoints")
EyeGaze = _message_class("EyeGaze")
ArmState = _message_class("ArmState")
ArmAction = _message_class("ArmAction")
GripperState = _message_class("GripperState")
Task = _message_class("Task")
Annotation = _message_class("Annotation")


def render_proto() -> str:
    """The `.proto` text for MESSAGES, as handed to vendors."""
    lines = [
        "// Generated by `python -m egomimic.rldb.mcap.schemas`; do not edit by hand.",
        "// EgoVerse raw data spec 0.1: message types not covered by foxglove schemas.",
        "",
        'syntax = "proto3";',
        "",
        f"package {PACKAGE};",
    ]
    for m in MESSAGES:
        lines.append("")
        lines += [f"// {line}" for line in m.doc.split("\n")]
        lines.append(f"message {m.name} {{")
        for e in m.enums:
            lines.append(f"  enum {e.name} {{")
            lines += [f"    {v} = {n};" for n, v in enumerate(e.values)]
            lines.append("  }")
        for f in m.fields:
            prefix = "repeated " if f.repeated else ""
            comment = f"  // {f.doc}" if f.doc else ""
            lines.append(f"  {prefix}{f.kind} {f.name} = {f.number};{comment}")
        lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(render_proto(), end="")
