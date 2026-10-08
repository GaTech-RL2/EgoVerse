"""Publish a positive source allowlist without task/goal/success implementations."""

import ast
import shutil
from pathlib import Path, PurePosixPath

from .common import digest, file_hash, write_json

WHOLE = (
    "controllers/osc.py",
    "controllers/base_controller.py",
    "controllers/config/osc_pose.json",
    "robots/single_arm.py",
    "robots/robot.py",
    "models/grippers/panda_gripper.py",
    "utils/transform_utils.py",
    "utils/control_utils.py",
)
EXCLUDED = [
    "task definitions",
    "BDDL/goal predicates",
    "success and reward functions",
    "expert demonstrations",
    "saved policies",
    "evaluation traces",
    "other trials",
]


def build_view(libero_root, robosuite_root, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in WHOLE:
        source = Path(robosuite_root) / name
        relative = "robosuite/" + name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries.append(
            {
                "path": relative,
                "source_sha256": file_hash(source),
                "view_sha256": file_hash(target),
                "projection": "entire_file",
            }
        )
    relative = "libero/libero/envs/env_wrapper.py"
    source = Path(libero_root) / relative
    text = source.read_text()
    lines, members = text.splitlines(keepends=True), []
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "ControlEnv":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name in (
                    "__init__",
                    "step",
                ):
                    members.append((member.lineno, member.end_lineno))
    if len(members) != 2:
        raise ValueError("upstream_projection_changed")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Audited source projection: ControlEnv constructor and step only.\nclass ControlEnv:\n"
        + "\n".join("".join(lines[a - 1 : b]) for a, b in members)
    )
    entries.append(
        {
            "path": relative,
            "source_sha256": file_hash(source),
            "view_sha256": file_hash(target),
            "projection": members,
        }
    )
    # Include exactly the upstream native proprioception activation, no object sensors.
    relative = "libero/libero/envs/bddl_base_domain.py"
    source = Path(libero_root) / relative
    lines = source.read_text().splitlines(keepends=True)
    selected = [
        (i + 1, line)
        for i, line in enumerate(lines)
        if line.strip() == 'observables["robot0_joint_pos"]._active = True'
    ]
    if len(selected) != 1:
        raise ValueError("native_sensor_activation_changed")
    target = destination / relative
    target.write_text(
        "# Audited native sensor activation excerpt; object/task code excluded.\n"
        + selected[0][1]
    )
    entries.append(
        {
            "path": relative,
            "source_sha256": file_hash(source),
            "view_sha256": file_hash(target),
            "projection": [[selected[0][0], selected[0][0]]],
        }
    )
    manifest = {"files": entries, "exclusions": EXCLUDED, "sha256": digest(entries)}
    write_json(destination / "allowlist.json", manifest)
    for path in destination.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return manifest


class SourceView:
    def __init__(self, root):
        self.root = Path(root).resolve()
        import json

        self.manifest = json.loads((self.root / "allowlist.json").read_text())
        self.allowed = {v["path"]: v["view_sha256"] for v in self.manifest["files"]}
        self.verify()

    def verify(self):
        for name, expected in self.allowed.items():
            path = self.root / name
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError("source_view_changed")

    def read(self, path, start=1, lines=160):
        if (
            type(path) is not str
            or path not in self.allowed
            or PurePosixPath(path).is_absolute()
        ):
            raise ValueError("source_not_allowlisted")
        if (
            type(start) is not int
            or type(lines) is not int
            or start < 1
            or not 1 <= lines <= 200
        ):
            raise ValueError("source_range")
        self.verify()
        content = (self.root / path).read_text().splitlines()
        return {
            "path": path,
            "start": start,
            "text": "\n".join(
                f"{i+start}: {s}"
                for i, s in enumerate(content[start - 1 : start - 1 + lines])
            ),
            "sha256": self.allowed[path],
        }

    def search(self, query):
        if type(query) is not str or not 1 <= len(query) <= 100:
            raise ValueError("source_query")
        self.verify()
        hits = []
        for name in sorted(self.allowed):
            for index, line in enumerate(
                (self.root / name).read_text().splitlines(), 1
            ):
                if query.casefold() in line.casefold():
                    hits.append({"path": name, "line": index, "text": line[:300]})
                    if len(hits) == 40:
                        return hits
        return hits
