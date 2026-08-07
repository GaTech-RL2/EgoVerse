"""
Smoke test for ZarrEpisodePackedDataset with chunking='none'.

Loads the pushT test_demos folder, runs it through a DataLoader with
pack_collate, then dumps:
  - PNG of the first / mid / last frame of each episode in the batch
  - PNG line plot of every action dim vs packed frame index, with vertical
    lines at cu_seqlens boundaries to mark per-episode segments

Outputs go to ./packed_smoke_out/.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from egomimic.rldb.zarr.zarr_dataset_packed import (
    ZarrEpisodePackedDataset,
    pack_collate,
)

KEY_MAP = {
    "front_img_1": {
        "key_type": "camera_keys",
        "zarr_key": "observations.images.front_img_1",
    },
    "state_agent_obj": {
        "key_type": "proprio_keys",
        "zarr_key": "observations.state",
    },
    "actions": {
        "key_type": "action_keys",
        "zarr_key": "actions",
    },
    "annotations": {
        "key_type": "annotation_keys",
        "zarr_key": "annotations",
    },
    "embodiment": {
        "key_type": "metadata_keys",
        "zarr_key": "metadata.embodiment",
    },
}

DATA_FOLDER = "/coc/cedarp-dxu345-0/Tsim_datasets2/circle"
OUT_DIR = Path("./packed_smoke_out")
N_EPISODES_TO_SAVE_VIDEOS = 2  # cap on per-episode MP4 dumps
VIDEO_FPS = 30


def _img_to_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [0, 1] -> (H, W, 3) uint8 for PIL."""
    arr = frame_chw.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = ZarrEpisodePackedDataset.from_local_folder(
        folder_path=DATA_FOLDER,
        key_map=KEY_MAP,
        max_seq_len=None,
        chunking="none",
    )
    print(f"# samples (one per full episode): {len(ds)}")
    print(f"# unique episodes: {len(ds.datasets)}")
    for i, (key, start, end) in enumerate(ds.index):
        print(f"  idx={i}  episode={key}  frames=[{start}, {end})  len={end - start}")

    # Pack all 4 episodes into a single batch.
    dl = DataLoader(
        ds,
        batch_size=len(ds),
        shuffle=False,
        collate_fn=pack_collate,
        num_workers=0,
    )
    batch = next(iter(dl))

    seq_lens = batch["seq_lens"].tolist()
    cu = batch["cu_seqlens"].tolist()
    actions = batch["actions"]
    images = batch["front_img_1"]
    state = batch["state_agent_obj"]

    print("\n--- packed batch ---")
    print(f"batch_size       = {batch['batch_size']}")
    print(f"seq_lens         = {seq_lens}")
    print(f"cu_seqlens       = {cu}")
    print(f"max_seq_len      = {batch['max_seq_len']}")
    print(
        f"actions.shape    = {tuple(actions.shape)}  (expected first dim = {sum(seq_lens)})"
    )
    print(f"front_img_1.shape= {tuple(images.shape)}")
    print(f"state.shape      = {tuple(state.shape)}")
    print(f"embodiment       = {batch['embodiment'].tolist()}")
    print(f"episode_idx      = {batch['episode_idx'].tolist()}")
    print(f"chunk_offset     = {batch['chunk_offset'].tolist()}")
    annotations = batch["annotations"]
    print(f"annotations      = {annotations}")

    assert actions.shape[0] == sum(
        seq_lens
    ), f"per-frame dim {actions.shape[0]} != sum(seq_lens) {sum(seq_lens)}"
    assert cu[0] == 0 and cu[-1] == sum(seq_lens), "cu_seqlens malformed"
    for i in range(len(seq_lens)):
        assert (
            cu[i + 1] - cu[i] == seq_lens[i]
        ), f"cu_seqlens segment {i} mismatch: {cu[i + 1] - cu[i]} vs {seq_lens[i]}"

    ep_keys = [ds.index[i][0] for i in range(len(ds))]

    # 1. Dump an MP4 per episode (full playback) for the first N_EPISODES_TO_SAVE_VIDEOS
    # episodes in the batch. The actions plot below still covers every
    # packed episode so boundary checks aren't lost.
    n_to_save = min(N_EPISODES_TO_SAVE_VIDEOS, batch["batch_size"])
    saved_videos: list[Path] = []
    for b in range(n_to_save):
        s, e = cu[b], cu[b + 1]
        ep_name = ep_keys[b]
        out_path = OUT_DIR / f"ep{b:02d}_{ep_name}_pack_t{s}_to_t{e - 1}.mp4"
        with imageio.get_writer(
            str(out_path),
            fps=VIDEO_FPS,
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            for t in range(s, e):
                writer.append_data(_img_to_uint8(images[t]))
        saved_videos.append(out_path)
        print(f"saved {e - s}-frame mp4 for ep {b} ({ep_name}) -> {out_path}")
    if batch["batch_size"] > n_to_save:
        print(
            f"(skipped video dumps for {batch['batch_size'] - n_to_save} additional episode(s) "
            f"— bump N_EPISODES_TO_SAVE_VIDEOS to dump more)"
        )

    # 2. Plot all action dims over packed frame index, marking episode boundaries.
    actions_np = actions.detach().cpu().numpy()
    n_frames, action_dim = actions_np.shape
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 2.4 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    t_axis = np.arange(n_frames)
    for d in range(action_dim):
        axes[d].plot(t_axis, actions_np[:, d], linewidth=0.8, color="tab:blue")
        for boundary in cu[1:-1]:
            axes[d].axvline(boundary, color="tab:red", linestyle="--", alpha=0.6)
        axes[d].set_ylabel(f"action[{d}]")
        axes[d].grid(True, alpha=0.3)
    # Annotate which packed frame range belongs to which episode under the bottom plot.
    for b in range(batch["batch_size"]):
        s, e = cu[b], cu[b + 1]
        mid = (s + e) // 2
        axes[-1].text(
            mid,
            axes[-1].get_ylim()[0],
            ep_keys[b],
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:red",
        )
    axes[-1].set_xlabel("packed frame index")
    fig.suptitle(
        f"Actions packed across {batch['batch_size']} episodes "
        f"(total {n_frames} frames; red dashes = cu_seqlens boundaries)"
    )
    fig.tight_layout()
    actions_plot = OUT_DIR / "actions_packed.png"
    fig.savefig(actions_plot, dpi=120)
    plt.close(fig)
    print(f"saved actions plot -> {actions_plot}")

    # 3. Cross-check: actions at packed boundaries should equal actions at
    # frame 0 and last frame of each episode, read directly from the per-episode
    # ZarrDataset.
    print("\n--- cross-check vs per-episode reads ---")
    sorted_keys = sorted(ds.datasets.keys())
    for b, key in enumerate(sorted_keys):
        ep_ds = ds.datasets[key]
        # Direct per-frame read of first and last action.
        first_direct = ep_ds.episode_reader.read(
            {KEY_MAP["actions"]["zarr_key"]: (0, 1)}
        )[KEY_MAP["actions"]["zarr_key"]][0]
        last_direct = ep_ds.episode_reader.read(
            {
                KEY_MAP["actions"]["zarr_key"]: (
                    ep_ds.total_frames - 1,
                    ep_ds.total_frames,
                )
            }
        )[KEY_MAP["actions"]["zarr_key"]][0]
        s, e = cu[b], cu[b + 1]
        first_packed = actions_np[s]
        last_packed = actions_np[e - 1]
        first_match = np.allclose(first_direct, first_packed, atol=1e-6)
        last_match = np.allclose(last_direct, last_packed, atol=1e-6)
        print(
            f"  {key}: first match={first_match} last match={last_match} | "
            f"direct first={first_direct.tolist()} packed first={first_packed.tolist()}"
        )
        assert first_match and last_match, f"action mismatch for {key}"

    print("\nAll checks passed.")
    print(f"See outputs in: {OUT_DIR.resolve()}")
    print("\n--- saved MP4s (absolute paths) ---")
    for vp in saved_videos:
        print(vp.resolve())


if __name__ == "__main__":
    main()
