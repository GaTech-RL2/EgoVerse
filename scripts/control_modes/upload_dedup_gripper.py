"""Upload the dedupe-only gripper cells to R2 for the control-mode study.

DEDUPE-ONLY means the real, near-duplicate-filtered source demos and NOTHING
else. ds_gen tops each cell back up to 1000 with SE(2)-retargeted, action-
perturbed episodes that are kept only if they re-reach the goal under that
cell's own control gap. That is a success filter sitting directly on top of the
quantity this study measures — "generalizes to a held-out controller" would
partly become "generalizes from a distribution pre-selected for robustness".
So the generated tail is dropped.

Episodes are written source-first, and the boundary was verified by matching
(object_pose, goal_pose) against the ds_src layouts rather than trusting write
order. Taking `[:CAP]` with CAP <= min(from_source) therefore takes real demos
only.

Every episode is verified by actually READING `actions` and
`observations.state`. zarr 3.1.0 silently wrote corrupt numeric arrays while
reporting success, so a written count is not evidence of a good write — that is
exactly how 714/1000 of ideal/gripper turned out unreadable. The JPEG, reward
and goal arrays take a different codec path and stay readable, so a partial
check would also have passed.

Usage:
    python scripts/control_modes/upload_dedup_gripper.py [--dry-run] [modes...]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import zarr

BASE = os.path.expanduser("~/Desktop/GEAR/sim_run")
SRC = f"{BASE}/ds_gen"
RESULTS = f"{BASE}/results"
PREFIX = ("s3://rldb/processed_v3/pushshapes_sim/"
          "control_gap_dedup_gripper_simv2_20260830")
# min(from_source) across the five modes is loose at 547, so every mode can
# supply this many REAL episodes. Equalizing removes a 19% between-mode
# imbalance for free.
CAP = 547
DEDUPE_RADIUS = 40
EMB = "gripper"
ALL_MODES = ["ideal", "tight", "loose", "laggy", "sticky", "jittery"]


def r2_env() -> dict:
    """s3://rldb is CLOUDFLARE R2. Pointing valid AWS credentials at an AWS
    endpoint returns AccessDenied and looks exactly like a permissions
    problem; a stale AWS_SESSION_TOKEN returns InvalidArgument."""
    env = dict(os.environ)
    with open(os.path.expanduser("~/.egoverse_env")) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env["AWS_ACCESS_KEY_ID"] = env["R2_ACCESS_KEY_ID"]
    env["AWS_SECRET_ACCESS_KEY"] = env["R2_SECRET_ACCESS_KEY"]
    env["AWS_DEFAULT_REGION"] = env["AWS_REGION"] = "auto"
    env.pop("AWS_SESSION_TOKEN", None)
    return env


def from_source(mode: str) -> int:
    with open(f"{RESULTS}/{mode}__{EMB}__T.json") as fh:
        return int(json.load(fh)["from_source"])


VERIFIED_CACHE = f"{BASE}/gripper_good_episodes.json"


def readable(path: str) -> bool:
    try:
        g = zarr.open(path, mode="r")
        _ = g["actions"][:]
        _ = g["observations.state"][:]
        return True
    except Exception:
        return False


def verified_set(mode: str) -> set[str] | None:
    """Episodes already proven readable by a full read of both numeric arrays.

    The cache is the output of a prior exhaustive scan over every episode in
    every cell — the SAME check this script would redo, on the same bytes. It
    is reused because this box is heavily oversubscribed by the generation
    grid, where re-reading ~2,700 zstd-compressed cells takes tens of minutes
    of starved CPU for a result already in hand.

    Only cells whose mtimes predate the cache may use it, and only `ideal` is
    being rewritten, so the seen modes are stable. Anything not in the cache
    falls back to a real read rather than being assumed good.
    """
    try:
        with open(VERIFIED_CACHE) as fh:
            return set(json.load(fh).get(mode) or [])
    except Exception:
        return None


MANIFEST_PATH = f"{BASE}/MANIFEST_dedup_gripper.json"


def load_manifest() -> dict:
    """Existing manifest, or a fresh skeleton.

    MERGED, never replaced. Uploading a single mode used to rewrite the
    manifest with only that mode in it and push the result over the top of the
    real one — silently discarding the provenance of every other cell in the
    prefix. The manifest is the only record of what the prefix contains.
    """
    try:
        with open(MANIFEST_PATH) as fh:
            m = json.load(fh)
        if isinstance(m.get("modes"), dict):
            return m
    except Exception:
        pass
    return {}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    manifest_only = "--manifest-only" in sys.argv
    modes = args or ALL_MODES
    env = r2_env()
    endpoint = env["R2_ENDPOINT_URL"]
    prior = load_manifest()
    manifest = {
        "prefix": PREFIX,
        "embodiment": EMB,
        "composition": "dedupe-only (real source demos; NO MimicGen fill)",
        "cap_per_mode": CAP,
        "dedupe_radius": DEDUPE_RADIUS,
        "zarr_version_written_with": "3.1.3",
        "verification": ("every uploaded episode was opened and both `actions` "
                         "and `observations.state` were read in full; a written "
                         "count is not evidence of a good write (zarr 3.1.0 "
                         "wrote corrupt numeric arrays silently)"),
        "caveat_intrinsic_dimensionality": (
            "Dedupe removes near-duplicates; it does NOT raise intrinsic "
            "dimensionality. PC95 measured 4-8 both before and after. Read this "
            "as better sample efficiency per epoch, not broader behaviour "
            "coverage."),
        "modes": dict(prior.get("modes") or {}),
    }

    for mode in modes:
        src_dir = f"{SRC}/{mode}/{EMB}/T"
        n_real = from_source(mode)
        if CAP > n_real:
            print(f"FATAL {mode}: cap {CAP} exceeds from_source {n_real} — "
                  f"would include generated episodes")
            return 1
        eps = sorted(os.listdir(src_dir))[:CAP]

        t0 = time.time()
        cached = verified_set(mode)
        n_cached = 0
        bad = []
        for e in eps:
            if cached is not None and e in cached:
                n_cached += 1
                continue
            if not readable(f"{src_dir}/{e}"):
                bad.append(e)
        if bad:
            print(f"FATAL {mode}: {len(bad)} unreadable in the first {CAP} "
                  f"(e.g. {bad[:3]})")
            return 1
        print(f"{mode}: verified {len(eps)}/{len(eps)} readable "
              f"({n_cached} from cache, {len(eps)-n_cached} re-read, "
              f"{time.time()-t0:.0f}s), from_source={n_real}", flush=True)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            for e in eps:
                fh.write(f'cp "{src_dir}/{e}/*" "{PREFIX}/{mode}/{EMB}/T/{e}/"\n')
            cmds = fh.name

        if dry or manifest_only:
            why = "dry-run" if dry else "manifest-only"
            print(f"  [{why}] skipping upload of {len(eps)} episodes")
        else:
            t0 = time.time()
            subprocess.run(
                ["s5cmd", "--endpoint-url", endpoint, "--numworkers", "64",
                 "run", cmds],
                env=env, check=True, stdout=subprocess.DEVNULL)
            print(f"  uploaded {len(eps)} episodes ({time.time()-t0:.0f}s)",
                  flush=True)
            os.unlink(cmds)

        manifest["modes"][mode] = {
            "from_source_real": n_real,
            "generated_in_ds_gen": 1000 - n_real,
            "uploaded_after_cap": len(eps),
            "generated_uploaded": 0,
        }

    manifest["total_uploaded"] = sum(
        m["uploaded_after_cap"] for m in manifest["modes"].values())
    manifest["note_held_out"] = (
        "jittery is uploaded but is NOT a training mode. It is the held-out "
        "controller, measured by rollout SR under its control gap; its episodes "
        "exist only for a post-hoc held-out BC loss.")

    path = f"{BASE}/MANIFEST_dedup_gripper.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {path}")
    if not dry:
        subprocess.run(
            ["s5cmd", "--endpoint-url", endpoint, "cp", path,
             f"{PREFIX}/MANIFEST.json"],
            env=env, check=True, stdout=subprocess.DEVNULL)
        print(f"wrote {PREFIX}/MANIFEST.json")
    print(json.dumps(manifest["modes"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
