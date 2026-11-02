"""Authenticated first-phase reuse still requires real verification and resume."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.fold_clothes_moving_opscale import run_preflight as preflight

COMMIT = "d" * 40
DESIGN = "a" * 64
OLD_JOB = "12345"
NEW_JOB = "67890"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evaluation(folder, operators, step=2):
    split, videos, coverage, tracking = {}, [], {}, {}
    for cohort, count, pins in [
        ("seen_matched", 40, 4),
        ("seen_perlevel", 3 * operators, 4 if operators == 4 else 5),
        ("unseen", 400, 5),
    ]:
        episodes = [f"{cohort}-{index}" for index in range(count)]
        split[cohort] = episodes
        split[f"video_{cohort}"] = episodes[:pins]
        coverage[cohort] = {"expected": count, "observed": count}
        tracking[cohort] = {
            "visited": count,
            "scored": count,
            "skipped": 0,
            "episodes": {
                eid: {"visited": 1, "scored": 1, "skipped": 0} for eid in episodes
            },
        }
        for episode in episodes[:pins]:
            path = folder / "0/videos" / f"{episode}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic video evidence")
            videos.append(
                {
                    "path": str(path),
                    "cohort": cohort,
                    "episode": episode,
                    "wandb_key": f"videos_open_loop/{cohort}/{episode}",
                    "evaluation_policy": preflight.EVALUATION_POLICY,
                    "tracking": {"visited": 1, "scored": 1, "skipped": 0},
                    "source_frames": 100,
                    "unscored_frames": 0,
                    "unscored_spans": [],
                    "timing": "original_source_frames_with_labeled_unscored_spans",
                }
            )
    return split, {
        "global_step": step,
        "epoch": step // 2 - 1,
        "videos": videos,
        "coverage": coverage,
        "evaluation_policy": preflight.EVALUATION_POLICY,
        "tracking_coverage": tracking,
        "metrics": {f"val/{cohort}/Valid/action_loss": 0.5 for cohort in coverage},
    }


@pytest.fixture
def prior(tmp_path):
    runs = tmp_path / "runs"
    folder = runs / "smoke" / COMMIT[:12] / "b24_op4"
    folder.mkdir(parents=True)
    run_id = f"fold_moving_smoke_b24_op4_{COMMIT[:10]}_{OLD_JOB}"
    args = SimpleNamespace(
        reuse_first_phase=runs / "operations/reuse.json", reuse_first_phase_sha256=None
    )
    overrides = [
        "trainer.max_epochs=1",
        "trainer.min_epochs=1",
        "trainer.limit_train_batches=2",
        "trainer.check_val_every_n_epoch=1",
        "callbacks.model_checkpoint.every_n_epochs=1",
        "callbacks.moving_opscale_evaluation.viz_every_n_epochs=1",
        f"norm_stats.precomputed_norm_path={runs / 'norm/b24_op4/norm_stats'}",
        f"logger.wandb.id={run_id}",
        "logger.wandb.group=fold_clothes_moving_opscale_smoke",
        "++logger.wandb.resume=never",
    ]
    files = {
        "epoch_1.log": "Finished synthetic first phase",
        "0/.hydra/config.yaml": "synthetic: true",
        "0/.hydra/overrides.yaml": json.dumps(overrides),
        "0/.hydra/hydra.yaml": "synthetic: true",
        "0/git_sha.txt": COMMIT,
        "0/git.diff": "",
        "0/git_untracked.txt": "",
        "0/checkpoints/epoch_epoch=0.ckpt": "synthetic checkpoint bytes",
    }
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    split, status = evaluation(folder, 4)
    save(folder / "0/evaluation_status.json", status)
    files["0/evaluation_status.json"] = ""
    for video in status["videos"]:
        files[str(Path(video["path"]).relative_to(folder))] = ""
    failure_path = runs / "operations/prior_failure.json"
    save(
        failure_path,
        {
            "status": "failed",
            "commit": COMMIT,
            "design_sha256": DESIGN,
            "error": "W&B smoke evidence did not become available: RuntimeError: W&B missing video: synthetic",
        },
    )
    receipt = {
        "schema_version": 1,
        "status": "ready_for_reverification",
        "commit": COMMIT,
        "design_sha256": DESIGN,
        "job_id": OLD_JOB,
        "run_id": run_id,
        "folder": str(folder),
        "failed_report": str(failure_path),
        "failed_report_sha256": preflight.sha(failure_path),
        "files": {name: preflight.sha(folder / name) for name in files},
    }

    def repin(*changed_files):
        for name in changed_files:
            receipt["files"][name] = preflight.sha(folder / name)
        save(args.reuse_first_phase, receipt)
        args.reuse_first_phase_sha256 = preflight.sha(args.reuse_first_phase)

    repin()
    return SimpleNamespace(
        args=args,
        runs=runs,
        folder=folder,
        receipt=receipt,
        repin=repin,
        split=split,
        status=status,
        run_id=run_id,
        checkpoint=folder / "0/checkpoints/epoch_epoch=0.ckpt",
    )


def authenticate(prior):
    return preflight.reusable_first_phase(prior.args, prior.runs, COMMIT, DESIGN)


def test_valid_receipt_only_authenticates_inputs_without_passing_gate(
    prior, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Authentication must not claim checkpoint/backend verification")

    monkeypatch.setattr(preflight, "inspect_checkpoint", forbidden)
    monkeypatch.setattr(preflight, "verify_wandb", forbidden)
    result = authenticate(prior)
    assert result["checkpoint"] == prior.checkpoint
    assert result["run_id"] == prior.run_id
    assert "status" not in result
    assert not (prior.runs / "evaluation_smoke.json").exists()
    assert not prior.args.reuse_first_phase.with_name(
        "first_phase_reverified.json"
    ).exists()


def test_receipt_must_be_pinned_and_unchanged(prior):
    checksum = prior.args.reuse_first_phase_sha256
    prior.args.reuse_first_phase_sha256 = None
    with pytest.raises(RuntimeError, match="must be paired"):
        authenticate(prior)
    prior.args.reuse_first_phase_sha256 = checksum
    with prior.args.reuse_first_phase.open("a") as stream:
        stream.write(" ")
    with pytest.raises(RuntimeError, match="receipt changed"):
        authenticate(prior)


def test_checkpoint_cannot_change_after_receipt(prior):
    prior.checkpoint.write_bytes(b"replacement checkpoint")
    with pytest.raises(RuntimeError, match="artifact changed"):
        authenticate(prior)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("commit", "e" * 40, "identity mismatch"),
        ("run_id", "another-run", "W&B identity mismatch"),
    ],
)
def test_identity_mismatch_is_rejected_even_when_receipt_is_repinned(
    prior, field, value, error
):
    prior.receipt[field] = value
    prior.repin()
    with pytest.raises(RuntimeError, match=error):
        authenticate(prior)


@pytest.mark.parametrize(
    "name,text,error",
    [
        ("0/git.diff", "source edit", "source was dirty"),
        ("0/git_untracked.txt", "extra.py", "untracked files"),
        ("0/git_sha.txt", "e" * 40, "training source differs"),
    ],
)
def test_dirty_or_wrong_training_source_cannot_be_reused(prior, name, text, error):
    (prior.folder / name).write_text(text)
    prior.repin(name)
    with pytest.raises(RuntimeError, match=error):
        authenticate(prior)


def test_missing_video_is_rejected(prior):
    prior.status["videos"].pop()
    save(prior.folder / "0/evaluation_status.json", prior.status)
    prior.repin("0/evaluation_status.json")
    with pytest.raises(RuntimeError, match="video evidence incomplete"):
        authenticate(prior)


def test_reuse_folder_and_artifacts_cannot_escape_smoke_root(prior, tmp_path):
    prior.receipt["folder"] = str(tmp_path / "outside")
    prior.repin()
    with pytest.raises(RuntimeError, match="outside this smoke root"):
        authenticate(prior)
    prior.receipt["folder"] = str(prior.folder)
    outside = prior.folder.parent / "outside.txt"
    outside.write_text("unrelated")
    prior.receipt["files"]["../outside.txt"] = preflight.sha(outside)
    prior.repin()
    with pytest.raises(RuntimeError, match="artifact path"):
        authenticate(prior)


def test_video_outside_authenticated_folder_is_rejected(prior, tmp_path):
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"unrelated")
    prior.status["videos"][0]["path"] = str(outside)
    save(prior.folder / "0/evaluation_status.json", prior.status)
    prior.repin("0/evaluation_status.json")
    with pytest.raises(RuntimeError, match="video is outside"):
        authenticate(prior)


def test_changed_recipe_overrides_are_rejected(prior):
    path = prior.folder / "0/.hydra/overrides.yaml"
    overrides = json.loads(path.read_text())
    overrides[2] = "trainer.limit_train_batches=1"
    save(path, overrides)
    prior.repin("0/.hydra/overrides.yaml")
    with pytest.raises(RuntimeError, match="overrides differ"):
        authenticate(prior)


def test_prior_epoch_two_attempt_is_never_overwritten(prior):
    (prior.folder / "epoch_2.log").write_text("preserve this failed attempt")
    with pytest.raises(RuntimeError, match="resume was already attempted"):
        authenticate(prior)


def test_smoke_reverifies_then_resumes_same_run_and_starts_op128_fresh(
    prior, tmp_path, monkeypatch
):
    private = tmp_path / "private"
    repo = tmp_path / "immutable-training-repo"
    here = repo / "experiments/fold_clothes_moving_opscale"
    save(
        private / "canonical_zarr/_episode_index.json",
        {
            f"{cohort}-{index}": {"total_frames": 100}
            for cohort in ("seen_matched", "seen_perlevel", "unseen")
            for index in range(400)
        },
    )
    for operators in [4, 128]:
        cell = f"b24_op{operators}"
        split = (
            prior.split
            if operators == 4
            else evaluation(tmp_path / "fixture-op128", 128)[0]
        )
        save(private / "splits" / f"fold_moving_{cell}.json", split)
        save(
            here / "budget_24" / f"op_{operators}.yaml",
            {"synthetic_operators": operators},
        )
        stats = prior.runs / "norm" / cell / "norm_stats/norm_stats.json"
        save(stats, {"synthetic": True})
        save(
            prior.runs / "norm" / cell / "norm_complete.json",
            {
                "status": "passed",
                "commit": COMMIT,
                "design_sha256": DESIGN,
                "stats_path": str(stats),
                "stats_sha256": preflight.sha(stats),
                "canonical_index_sha256": preflight.sha(
                    private / "canonical_zarr/_episode_index.json"
                ),
                "split_sha256": preflight.sha(
                    private / "splits" / f"fold_moving_{cell}.json"
                ),
            },
        )
    args = prior.args
    args.training_repo = repo
    args.output = prior.runs / "evaluation_smoke.json"
    args.operational_controller = {"commit": "c" * 40}
    monkeypatch.setattr(
        preflight, "context", lambda unused: (private, prior.runs, COMMIT, DESIGN)
    )
    monkeypatch.setenv("SLURM_JOB_ID", NEW_JOB)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda index: "NVIDIA H200",
            )
        ),
    )
    events = []

    def inspect(path):
        events.append(("inspect", Path(path)))
        step = (
            4
            if Path(path).name == "last.ckpt"
            and Path(path).is_relative_to(prior.folder)
            else 2
        )
        return {
            "step": step,
            "optimizer_state": True,
            "scheduler_state": True,
            "path": str(path),
        }

    def verify(run_id, status, expected_step):
        events.append(("verify", run_id, expected_step))
        return {
            "run_id": run_id,
            "state": "finished",
            "verified_videos": len(status["videos"]),
        }

    def execute(command, cwd, stdout, stderr):
        assert cwd == repo.resolve()
        assert str(repo / "egomimic/trainHydra.py") in command
        assert f"--config-dir={here / 'budget_24'}" in command
        options = dict(item.split("=", 1) for item in command if "=" in item)
        operators = int(options["--config-name"].removeprefix("op_"))
        epoch = int(options["trainer.max_epochs"])
        events.append(("execute", operators, epoch))
        folder = Path(options["hydra.sweep.dir"])
        if operators == 4:
            assert events[:2] == [
                ("inspect", prior.checkpoint),
                ("verify", prior.run_id, 2),
            ]
            assert epoch == 2
            assert options["logger.wandb.id"] == prior.run_id
            assert options["ckpt_path"] == str(prior.checkpoint)
            assert options["++logger.wandb.resume"] == "allow"
            assert prior.args.reuse_first_phase.with_name(
                "first_phase_reverified.json"
            ).exists()
            stdout.write("Restored all states\n")
        else:
            assert operators == 128 and epoch == 1
            assert (
                options["logger.wandb.id"]
                == f"fold_moving_smoke_b24_op128_{COMMIT[:10]}_{NEW_JOB}"
            )
            assert "ckpt_path" not in options
            assert options["++logger.wandb.resume"] == "never"
        checkpoint = folder / "0/checkpoints/last.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"new synthetic checkpoint")
        _, status = evaluation(folder, operators, step=epoch * 2)
        save(folder / "0/evaluation_status.json", status)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight, "inspect_checkpoint", inspect)
    monkeypatch.setattr(preflight, "verify_wandb", verify)
    monkeypatch.setattr(preflight.subprocess, "run", execute)
    preflight.smoke(args)
    assert [event for event in events if event[0] == "execute"] == [
        ("execute", 4, 2),
        ("execute", 128, 1),
    ]
    report = json.loads(args.output.read_text())
    assert report["status"] == "passed"
    assert len(report["runs"]) == 3
    assert report["commit"] == COMMIT
    assert report["operational_controller"] == args.operational_controller
    assert report["runs"][0]["execution_job_id"] == OLD_JOB
    assert report["runs"][0]["verification_job_id"] == NEW_JOB
    assert report["runs"][1]["checkpoint"]["resumed_from_step"] == 2
    assert report["runs"][1]["checkpoint"]["step"] == 4
    assert all(report["checks"].values())
