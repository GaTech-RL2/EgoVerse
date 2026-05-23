"""PCA on H-Net trunk tokens for cotrain runs — stage-agnostic.

This script does not assume a fixed number of stages, a particular
arrangement of ``PerEmbodimentStage`` wrappers, or balanced depths between
embodiments. It walks the live stage chain at runtime, finds the terminal
stage (the deepest stage that has no further ``inner_stage``), and hooks
its forward output. Whatever sits at the deepest point of whichever
embodiment-specific sub-chain is currently being executed gets captured.

For the standard cotrain setup (per-emb outer + shared inner trunk), the
terminal is the shared ``ComputeStage`` — same for both embodiments. For
asymmetric / imbalanced configs (e.g. one emb has 3 outer stages, another
has 1), the script still captures the deepest per-emb tokens because the
hook fires on whichever stage is the actual terminal at execution time.

Output:
  - <out_dir>/pca_2d.png — 2D PCA scatter, one trajectory per episode,
    colored by embodiment.
  - <out_dir>/pca_3d.png — 3D PCA scatter (same data, top-3 components).
  - <out_dir>/tokens.npz — raw trunk tokens + per-episode embodiment +
    seq_lens (so you can re-plot without re-running the model).

Usage (via Hydra, mirrors trainHydra's mode=eval pattern):
    python scripts/pca_trunk.py \\
        --config-name=train_zarr_cartesian \\
        mode=eval \\
        data=tsimulation_cotrain \\
        model=hnet_pushshapes_cotrain \\
        callbacks=null \\
        evaluator=eval_hnet \\
        ckpt_path=/path/to/last.ckpt \\
        +pca.out_dir=/path/to/out \\
        +pca.max_episodes_per_emb=16
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA

from egomimic.models.hnet_nets.stages import _BaseStage

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stage-chain walking
# --------------------------------------------------------------------------- #


def find_terminal_stages(root: _BaseStage) -> list[_BaseStage]:
    """Walk the chain from ``root`` and return every terminal stage (one with
    no ``inner_stage``).

    For a plain linear chain there's exactly one terminal. For wrappers like
    ``PerEmbodimentStage`` we descend into ``sub_stages``; if all sub-stages
    share the same inner trunk (the common cotrain case) we end up with one
    terminal. If sub-stages have independently-terminating sub-chains, we
    return all of them so we can hook each.
    """
    seen: set[int] = set()
    terminals: list[_BaseStage] = []

    def walk(stage: _BaseStage) -> None:
        if id(stage) in seen:
            return
        seen.add(id(stage))
        # PerEmbodimentStage exposes inner_stage as a property that mirrors
        # one sub-stage's inner pointer; walking via sub_stages directly is
        # more robust to wrappers that override inner_stage differently.
        if hasattr(stage, "sub_stages") and stage.sub_stages:
            for sub in stage.sub_stages.values():
                walk(sub)
            return
        if stage.inner_stage is None:
            terminals.append(stage)
            return
        walk(stage.inner_stage)

    walk(root)
    if not terminals:
        raise RuntimeError("No terminal stage found in chain.")
    return terminals


# --------------------------------------------------------------------------- #
# Token capture
# --------------------------------------------------------------------------- #


class _TrunkCapture:
    """Forward-hook collector. Captures (output_tokens, embodiment_id) per
    call so we can associate each token batch with its embodiment.

    Works for both padded ``(B, T, D)`` and packed ``(T_total, D)`` outputs.
    For packed, ``ctx.cu_seqlens`` is used to split into per-episode chunks.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        self._current_emb: Optional[str] = None
        self._current_cu: Optional[torch.Tensor] = None

    def begin_call(self, emb_id: Optional[str], cu_seqlens: Optional[torch.Tensor]) -> None:
        self._current_emb = emb_id
        self._current_cu = cu_seqlens

    def hook(self, _module, _inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return
        tokens = output.detach().to("cpu").float().numpy()
        self.records.append(
            {
                "tokens": tokens,
                "embodiment_id": self._current_emb,
                "cu_seqlens": (
                    self._current_cu.detach().to("cpu").long().numpy()
                    if self._current_cu is not None
                    else None
                ),
            }
        )


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #


@torch.no_grad()
def collect_trunk_tokens(
    cfg: DictConfig, max_episodes_per_emb: int = 16
) -> tuple[list[np.ndarray], list[str]]:
    """Build the model from cfg, load ckpt, run packed forward over val
    batches, capture trunk-stage output per-episode.

    Returns:
        episodes:  list of (T_trunk_episode, D_trunk) arrays.
        emb_names: list[str] same length, embodiment name per episode.
    """
    # ---- Build model + datamodule (mirrors trainHydra's eval mode) ----
    norm_stats = None  # built lazily by ModelWrapper from cfg.data
    datamodule = hydra.utils.instantiate(cfg.data, _convert_="partial")
    if hasattr(datamodule, "setup"):
        datamodule.setup(stage="validate")
    # ModelWrapper handles norm_stats / instantiation internally.
    model_wrapper = hydra.utils.instantiate(
        cfg.model, _convert_="partial", _recursive_=False
    )
    if hasattr(model_wrapper, "_instantiate_model"):
        model_wrapper.model = model_wrapper._instantiate_model(cfg.model, norm_stats)

    # ---- Load ckpt weights ----
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path is required.")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_wrapper.load_state_dict(state["state_dict"], strict=False)
    logger.info("Loaded weights from %s", ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper.to(device).eval()
    algo = model_wrapper.model
    policy = algo.nets["policy"]
    hnet = policy.hnet

    # ---- Hook every terminal stage's forward ----
    terminals = find_terminal_stages(hnet.stages[0])
    logger.info("Hooking %d terminal stage(s).", len(terminals))
    capture = _TrunkCapture()
    handles = [t.register_forward_hook(capture.hook) for t in terminals]

    # ---- Walk val loader ----
    val_loader = datamodule.val_dataloader()
    if hasattr(val_loader, "iterables"):
        iterables = val_loader.iterables
    else:
        iterables = {"default": val_loader}

    per_emb_count: dict[str, int] = {emb: 0 for emb in iterables}
    episodes: list[np.ndarray] = []
    emb_names: list[str] = []

    domain_by_id = algo.domain_by_id

    for emb_name, loader in iterables.items():
        if per_emb_count[emb_name] >= max_episodes_per_emb:
            continue
        for batch in loader:
            if per_emb_count[emb_name] >= max_episodes_per_emb:
                break
            # process_batch_for_training normalizes + moves to device.
            processed = algo.process_batch_for_training(batch)  # dict[emb_id -> batch]
            for emb_id, _b in processed.items():
                if domain_by_id.get(emb_id) != emb_name:
                    continue
                ac_key = algo.resolved_ac_keys[emb_id]
                actions = _b[ac_key]
                obs = algo._build_obs(_b, emb_id)
                cu = _b["cu_seqlens"]
                max_seqlen = int(_b["max_seq_len"])
                capture.begin_call(
                    emb_id=domain_by_id.get(emb_id), cu_seqlens=cu
                )
                policy.forward_packed(
                    actions, obs, cu, max_seqlen, embodiment_id=domain_by_id.get(emb_id)
                )
                # Split the latest captured tokens into per-episode chunks
                # using the cu_seqlens we recorded. NOTE: trunk tokens
                # don't have the same length as input tokens (post-chunking).
                # We slice proportionally: each input subseq [s,e) gets a
                # proportional slice of the trunk output, anchored by the
                # ratio of trunk_T_total to input_T_total.
                rec = capture.records[-1]
                trunk_tokens = rec["tokens"]
                if trunk_tokens.ndim == 2:
                    T_trunk = trunk_tokens.shape[0]
                else:
                    # Padded (B, T, D) — flatten by stripping pad via cu_seqlens.
                    T_trunk = trunk_tokens.shape[1]
                T_total = int(cu[-1].item())
                ratio = T_trunk / max(T_total, 1)
                cu_np = rec["cu_seqlens"]
                for b in range(len(cu_np) - 1):
                    if per_emb_count[emb_name] >= max_episodes_per_emb:
                        break
                    s_in, e_in = int(cu_np[b]), int(cu_np[b + 1])
                    s_t = int(round(s_in * ratio))
                    e_t = int(round(e_in * ratio))
                    if e_t <= s_t:
                        continue
                    if trunk_tokens.ndim == 2:
                        ep_tokens = trunk_tokens[s_t:e_t]
                    else:
                        ep_tokens = trunk_tokens[b, : e_t - s_t]
                    episodes.append(ep_tokens)
                    emb_names.append(emb_name)
                    per_emb_count[emb_name] += 1

    for h in handles:
        h.remove()

    return episodes, emb_names


# --------------------------------------------------------------------------- #
# PCA + plotting
# --------------------------------------------------------------------------- #


def fit_pca_and_plot(
    episodes: list[np.ndarray], emb_names: list[str], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tokens = np.concatenate(episodes, axis=0)
    print(
        f"Fitting PCA on {len(episodes)} episodes "
        f"({all_tokens.shape[0]} tokens × {all_tokens.shape[1]}d)"
    )

    # Fit one PCA on the union so episodes are in a comparable space.
    pca3 = PCA(n_components=3)
    projected = pca3.fit_transform(all_tokens)

    # Split back per-episode.
    splits = np.cumsum([len(e) for e in episodes[:-1]])
    per_ep = np.split(projected, splits)

    np.savez(
        out_dir / "tokens.npz",
        projected=projected,
        seq_lens=np.array([len(e) for e in episodes]),
        emb_names=np.array(emb_names),
        explained_variance_ratio=pca3.explained_variance_ratio_,
    )
    print(f"explained_variance_ratio = {pca3.explained_variance_ratio_}")

    # Color by emb.
    unique_embs = sorted(set(emb_names))
    cmap = plt.get_cmap("tab10")
    color_by_emb = {emb: cmap(i) for i, emb in enumerate(unique_embs)}

    # 2D
    fig, ax = plt.subplots(figsize=(8, 8))
    for ep, emb in zip(per_ep, emb_names):
        if ep.shape[0] < 2:
            continue
        ax.plot(
            ep[:, 0], ep[:, 1], color=color_by_emb[emb], alpha=0.55, linewidth=0.9
        )
        ax.scatter(ep[0, 0], ep[0, 1], color=color_by_emb[emb], s=18, marker="o")
        ax.scatter(ep[-1, 0], ep[-1, 1], color=color_by_emb[emb], s=18, marker="x")
    for emb, color in color_by_emb.items():
        ax.plot([], [], color=color, label=emb)
    ax.legend(loc="upper right")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"H-Net trunk tokens — PCA (○ = ep start, × = ep end)\n"
        f"{len(episodes)} eps, {all_tokens.shape[0]} tokens, "
        f"d={all_tokens.shape[1]}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pca_2d.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {out_dir / 'pca_2d.png'}")

    # 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for ep, emb in zip(per_ep, emb_names):
        if ep.shape[0] < 2:
            continue
        ax.plot(ep[:, 0], ep[:, 1], ep[:, 2], color=color_by_emb[emb], alpha=0.55)
        ax.scatter(ep[0, 0], ep[0, 1], ep[0, 2], color=color_by_emb[emb], s=18, marker="o")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_3d.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {out_dir / 'pca_3d.png'}")


# --------------------------------------------------------------------------- #
# Hydra entrypoint
# --------------------------------------------------------------------------- #


@hydra.main(version_base="1.3", config_path="../egomimic/hydra_configs")
def main(cfg: DictConfig) -> None:
    pca_cfg = cfg.get("pca", {})
    out_dir = Path(
        pca_cfg.get("out_dir", "/tmp/pca_trunk_out")
    )
    max_per_emb = int(pca_cfg.get("max_episodes_per_emb", 16))

    OmegaConf.set_struct(cfg, False)
    cfg.mode = "eval"  # force eval mode
    OmegaConf.set_struct(cfg, True)

    episodes, emb_names = collect_trunk_tokens(cfg, max_episodes_per_emb=max_per_emb)
    if not episodes:
        raise RuntimeError("Captured zero episodes. Check ckpt/data wiring.")
    fit_pca_and_plot(episodes, emb_names, out_dir)


if __name__ == "__main__":
    main()
