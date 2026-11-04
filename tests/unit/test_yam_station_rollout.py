"""A 2-step Eva HPT checkpoint trained here, through the Yam station's EgoVerse
policy and ``rollout --check-model`` on the station profile. CPU only; no
device is opened."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from fixtures.recipes import (
    RECIPES,
    common_overrides,
    cpu_trainer_overrides,
    hpt_small_overrides,
)
from fixtures.train_harness import compose_recipe, hermetic_env, write_fixtures

import egomimic
import egomimic.trainHydra as train_hydra
from egomimic.robot.yam_station.egoverse_policy import load_egoverse_policy

REPO = Path(egomimic.__file__).parents[1]
CONFIG = (
    Path(egomimic.__file__).parent / "hydra_configs/robot/yam_rl2_egoverse_rollout.yaml"
)


def _checkpoint(tmp_path, monkeypatch):
    hermetic_env(monkeypatch)
    recipe = RECIPES[("eva", "hpt")]
    data, out, hashes = write_fixtures(tmp_path, "eva")
    cfg = compose_recipe(
        recipe,
        common_overrides(
            recipe.embodiment,
            data,
            out,
            batch_size=2,
            num_workers=0,
            episode_hashes=hashes,
        )
        + cpu_trainer_overrides(2)
        + hpt_small_overrides(recipe.embodiment),
        out,
    )
    _, objects = train_hydra.train(cfg)
    path = tmp_path / "models" / "eva_hpt.ckpt"
    path.parent.mkdir()
    objects["trainer"].save_checkpoint(path)
    return path


def _obs():
    rng = np.random.default_rng(0)
    pose = np.r_[0.3, 0.1, 0.2, 0.1, -0.2, 0.3]
    obs = {
        "ee_poses": np.r_[pose, 0.5, pose + 0.01, 0.5],
        "joint_positions": np.r_[np.zeros(6), 0.5, np.zeros(6), 0.5],
    }
    for cam in ("front_img_1", "left_wrist_img", "right_wrist_img"):
        obs[cam] = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    return obs


def test_egoverse_checkpoint_rolls_out(tmp_path, monkeypatch):
    ckpt = _checkpoint(tmp_path, monkeypatch)
    profile = yaml.safe_load(CONFIG.read_text())
    policy = load_egoverse_policy(
        {
            **{k: v for k, v in profile["policy"].items() if k != "kind"},
            "checkpoint": str(ckpt),
            "device": "cpu",
        }
    )
    # A 2-step model's grippers are untrained: this checks plumbing, not quality.
    policy.gripper_clip_tolerance = 1e9
    plan = policy.predict(_obs())
    assert plan.shape == (policy.action_shape[0], 14) and np.isfinite(plan).all()
    # Identity base_T_model and zero-rotation 6D rows land on the model-frame
    # positions.
    policy.base_T_model = {a: np.eye(4) for a in policy.base_T_model}
    row = np.r_[
        0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.4, 0.5, 0.6, 0.7, 1, 0, 0, 0, 1, 0, 0.9
    ]
    expected = np.r_[0.1, 0.2, 0.3, 0, 0, 0, 0.4, 0.5, 0.6, 0.7, 0, 0, 0, 0.9]
    np.testing.assert_allclose(
        policy._actions(row[None], _obs())[0], expected, atol=1e-9
    )

    profile["model_browser"]["root"] = str(ckpt.parent)
    profile["policy"]["checkpoint"] = str(ckpt)
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "egomimic.robot.yam_station.rollout",
            "--config",
            str(tmp_path / "profile.yaml"),
            "--check-model",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["status"] == "passed" and report["hardware_opened"] is False
