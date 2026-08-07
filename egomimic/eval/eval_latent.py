import csv
import glob
import logging
import os
import shutil
import time
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment

logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str):
    """Log wall-clock duration of a code block at INFO level."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info("[timing] %s: %.2fs", label, dt)


class PILatentEvalVideo(EvalVideo):
    """
    PI-model evaluator that captures per-layer attention `key` states (output
    of each transformer block's `self_attn.k_proj`) and writes per-token CSVs
    + scatter PNGs colored by embodiment.

    Per PaliGemma layer, three slices are emitted:
        - <layer>_img       (image-token rows only)
        - <layer>_lang      (language-token rows only)
        - <layer>_combined  (image + language together; when emit_combined=True)

    Per expert layer, one slice (<layer>) — expert sees only action tokens.

    For each slice, controlled by individual flags:
        compute_umap     -> writes <layer>.png   (UMAP 3D, GPU via cuML)
        compute_tsne_2d  -> writes <layer>_tsne2d.png  (t-SNE 2D, GPU via cuML)
        compute_tsne_3d  -> writes <layer>_tsne.png    (t-SNE 3D, sklearn CPU; slow)
    """

    def __init__(
        self,
        save_plots: bool = True,
        limit_val_batches: int = 400,
        compute_umap: bool = True,
        compute_tsne_2d: bool = True,
        compute_tsne_3d: bool = False,
        compute_pca: bool = False,
        compute_pca_umap: bool = True,
        pca_n_components: int = 50,
        pca_for_downstream: bool = False,
        emit_combined: bool = True,
        color_by: str = "embodiment",  # "embodiment" or "hash"
    ):
        super().__init__(limit_val_batches=limit_val_batches)
        self.compute_umap = compute_umap
        self.compute_tsne_2d = compute_tsne_2d
        self.compute_tsne_3d = compute_tsne_3d
        # PCA controls.
        #   compute_pca       -> save first-3-PCs as pca_x/y/z columns
        #                        (and log cumulative explained variance ratio).
        #   compute_pca_umap  -> independently fit PCA-N then UMAP on those
        #                        N-dim features; saves as pca_umap_x/y/z
        #                        columns (denoised UMAP, often tighter
        #                        clusters than raw-UMAP).
        #   pca_for_downstream -> when True, the regular `compute_umap`/
        #                        compute_tsne_2d also consume PCA features
        #                        instead of raw keys (overrides what
        #                        umap_x/y/z and tsne2d_x/y mean!). Default
        #                        False so UMAP-on-raw and PCA-then-UMAP are
        #                        independent reductions.
        self.compute_pca = compute_pca
        self.compute_pca_umap = compute_pca_umap
        self.pca_n_components = pca_n_components
        self.pca_for_downstream = pca_for_downstream
        # When True, paligemma layers also emit a "<layer>_combined" slice
        # containing image + language tokens together (in addition to the
        # separate _img / _lang slices).
        self.emit_combined = emit_combined
        # Plot scatter color choice. "embodiment" (default) → one color per
        # embodiment (good for cotrain_pi_latent_random). "hash" → one color
        # per episode hash (good for cotrain_pi_latent_pairs where you want
        # to see specific episodes apart).
        if color_by not in ("embodiment", "hash"):
            raise ValueError(
                f"color_by must be 'embodiment' or 'hash', got {color_by!r}"
            )
        self.color_by = color_by
        self._layer_keys = {}  # layer_name -> list[np.ndarray (B, S, D)]
        self._row_hashes = []  # one entry per sample (replicated by S at write time)
        self._row_embodiments = []
        self._hook_handles = []
        # per-step buffer; each layer is recorded only on its FIRST forward
        # call within a capture window (prefix pass for PaliGemma, first
        # denoise step for the action expert).
        self._step_capture = {}
        self._capture_active = False
        self._n_rows = 0
        # Filled by the embed_prefix wrapper so paligemma hooks know where
        # to slice image tokens vs language tokens.
        self._n_img_tokens = None
        self._n_lang_tokens = None
        self._orig_embed_prefix = None
        self.save_plots = save_plots

    def latent_dir(self):
        return os.path.join(self.root_dir(), "latents")

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------
    def _iter_attn_layers(self):
        algo = self.model
        pi_model = algo.nets["policy"]
        paligemma = pi_model.paligemma_with_expert.paligemma.language_model
        expert = pi_model.paligemma_with_expert.gemma_expert.model
        for idx, layer in enumerate(paligemma.layers):
            yield f"paligemma_layer_{idx:02d}", layer.self_attn.k_proj, True
        for idx, layer in enumerate(expert.layers):
            yield f"expert_layer_{idx:02d}", layer.self_attn.k_proj, False

    def _register_hooks(self):
        self._hook_handles = []

        # Wrap embed_prefix so we know the image/language token boundary for
        # the paligemma prefix pass. Prefix layout is [img_tokens, lang_tokens].
        pi_model = self.model.nets["policy"]
        self._orig_embed_prefix = pi_model.embed_prefix

        def wrapped_embed_prefix(images, img_masks, lang_tokens, lang_masks):
            embs, pad, att = self._orig_embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            if self._capture_active:
                self._n_lang_tokens = int(lang_masks.shape[1])
                self._n_img_tokens = int(embs.shape[1]) - self._n_lang_tokens
            return embs, pad, att

        pi_model.embed_prefix = wrapped_embed_prefix

        def make_hook(layer_name, is_paligemma):
            def _hook(_module, _inp, out):
                if not self._capture_active:
                    return
                if is_paligemma:
                    img_key = f"{layer_name}_img"
                    lang_key = f"{layer_name}_lang"
                    combined_key = f"{layer_name}_combined"
                    already = (
                        img_key in self._step_capture and lang_key in self._step_capture
                    )
                    if self.emit_combined:
                        already = already and combined_key in self._step_capture
                    if already:
                        return
                    if self._n_img_tokens is None or self._n_lang_tokens is None:
                        return
                    n_img = self._n_img_tokens
                    self._step_capture[img_key] = out[:, :n_img].detach()
                    self._step_capture[lang_key] = out[:, n_img:].detach()
                    if self.emit_combined:
                        self._step_capture[combined_key] = out.detach()
                else:
                    if layer_name in self._step_capture:
                        return
                    self._step_capture[layer_name] = out.detach()

            return _hook

        for layer_name, k_proj, is_paligemma in self._iter_attn_layers():
            self._hook_handles.append(
                k_proj.register_forward_hook(make_hook(layer_name, is_paligemma))
            )

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        if self._orig_embed_prefix is not None:
            self.model.nets["policy"].embed_prefix = self._orig_embed_prefix
            self._orig_embed_prefix = None

    # ------------------------------------------------------------------
    # Hash extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_hashes(_batch, batch_size, embodiment_name):
        val = _batch.get("episode_hash")
        if val is None:
            return [f"{embodiment_name}_unknown"] * batch_size
        if isinstance(val, (list, tuple)):
            return [str(v) for v in val]
        if isinstance(val, np.ndarray):
            return [str(v) for v in val.tolist()]
        if torch.is_tensor(val):
            return [str(v) for v in val.cpu().tolist()]
        return [str(val)] * batch_size

    # ------------------------------------------------------------------
    # Validation lifecycle
    # ------------------------------------------------------------------
    def on_validation_start(self):
        super().on_validation_start()
        self._layer_keys = {}
        self._row_hashes = []
        self._row_embodiments = []
        self._n_rows = 0
        self._register_hooks()
        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.latent_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics = {}
        images_dict = {}
        unnorm_preds = {}
        mse = MeanSquaredError()

        with torch.no_grad():
            for embodiment_id, _batch in batch.items():
                embodiment_name = get_embodiment(embodiment_id).lower()
                ac_key = algo.ac_keys[embodiment_id]
                proprio_keys = algo.proprio_keys[embodiment_id]
                lang_keys = algo.lang_keys[embodiment_id]
                camera_keys = algo.camera_keys.get(embodiment_id, algo.pi_cam_keys)

                processed_obs, _action = algo._robomimic_to_pi_data(
                    _batch,
                    camera_keys,
                    proprio_keys,
                    lang_keys,
                    ac_key,
                    embodiment_name,
                )

                # Open a capture window around a single forward per embodiment.
                self._step_capture = {}
                self._n_img_tokens = None
                self._n_lang_tokens = None
                self._capture_active = True
                pred_actions = algo.nets["policy"].sample_actions(
                    device=algo.device,
                    observation=processed_obs,
                    noise=None,
                    num_steps=algo.num_steps,
                )
                self._capture_active = False

                # Mirror PI's post-processing for metrics + viz.
                ref = _batch[ac_key]
                B, T, D = ref.shape
                converter = algo.action_registry.get(embodiment_id, ac_key)
                pred_actions_orig = converter.from32(pred_actions)
                pred = pred_actions_orig[:, :T, :D]

                predictions = OrderedDict()
                predictions[ac_key] = pred
                unnorm_actions = algo.norm_stats.unnormalize(predictions, embodiment_id)
                for k in unnorm_actions:
                    unnorm_preds[f"{embodiment_name}_{k}"] = unnorm_actions[k]

                unnorm_batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
                pred_key = f"{embodiment_name}_{ac_key}"
                if pred_key in unnorm_preds:
                    metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                        unnorm_preds[pred_key].cpu(), unnorm_batch[ac_key].cpu()
                    )
                    metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                        unnorm_preds[pred_key][:, -1].cpu(),
                        unnorm_batch[ac_key][:, -1].cpu(),
                    )

                if self.viz_func is not None:
                    images_dict[embodiment_id] = self._visualize_preds(
                        unnorm_preds, unnorm_batch
                    )

                hashes = self._extract_hashes(_batch, B, embodiment_name)
                self._row_hashes.extend(hashes)
                self._row_embodiments.extend([embodiment_name] * B)
                for layer_name, key_tensor in self._step_capture.items():
                    keys_bsd = key_tensor.to(torch.float32).cpu().numpy()
                    self._layer_keys.setdefault(layer_name, []).append(keys_bsd)
                self._n_rows += B

        return metrics, images_dict

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            return {}  # viz_func not configured — skip action visualization
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        return self.viz_func[embodiment_name](predictions, batch)

    def on_validation_end(self):
        super().on_validation_end()
        self._remove_hooks()

        if not self.trainer.is_global_zero:
            return
        if self._n_rows == 0:
            logger.warning(
                "PILatentEvalVideo: no latents captured; skipping CSV write."
            )
            return

        out_dir = os.path.join(self.latent_dir(), f"epoch_{self.trainer.current_epoch}")
        os.makedirs(out_dir, exist_ok=True)

        total_t0 = time.perf_counter()
        for layer_name, chunks in self._layer_keys.items():
            layer_t0 = time.perf_counter()
            # Each chunk is (b_i, S, D). Concat along batch, then flatten the
            # sequence axis so every token becomes its own CSV row.
            with _timed(f"{layer_name} | concat+reshape"):
                keys_bsd = np.concatenate(chunks, axis=0)  # (N, S, D)
                N, S, D = keys_bsd.shape
                keys = keys_bsd.reshape(N * S, D)
                sample_hashes = self._row_hashes
                sample_embs = self._row_embodiments
                if N != len(sample_hashes):
                    n = min(N, len(sample_hashes))
                    logger.warning(
                        "Sample-count mismatch for %s: keys=%d hashes=%d; truncating to %d.",
                        layer_name,
                        N,
                        len(sample_hashes),
                        n,
                    )
                    keys = keys_bsd[:n].reshape(n * S, D)
                    sample_hashes = sample_hashes[:n]
                    sample_embs = sample_embs[:n]
                # Replicate per-sample metadata across the S tokens.
                # frame_idx is the per-run sample index (0..N-1) — tokens of
                # the same frame share it, so meanpool.py can group on
                # (video_hash, frame_idx). token_idx is the position within
                # the sequence, useful for token-type slicing later.
                hashes = [h for h in sample_hashes for _ in range(S)]
                embs = [e for e in sample_embs for _ in range(S)]
                frame_idx = [i for i in range(len(sample_hashes)) for _ in range(S)]
                token_idx = list(range(S)) * len(sample_hashes)

            # Optional PCA: fit on raw `keys`, log explained variance, and
            # PCA: fit once if any PCA-dependent reduction (compute_pca for
            # the pca_x/y/z columns, compute_pca_umap for pca_umap_x/y/z, or
            # pca_for_downstream which routes UMAP/t-SNE through PCA features).
            need_pca = (
                self.compute_pca or self.compute_pca_umap or self.pca_for_downstream
            )
            pca_features = None
            pca_xyz = None
            if need_pca:
                with _timed(
                    f"{layer_name} | PCA-{self.pca_n_components} ({keys.shape[0]} rows)"
                ):
                    pca_features, evr = self._pca(keys, self.pca_n_components)
                cum = float(np.cumsum(evr)[-1]) if evr.size else 0.0
                logger.info(
                    "[PCA] %s: %d components -> cumulative explained variance = %.2f%% "
                    "(target ~70-80%%; ratios head: %s)",
                    layer_name,
                    pca_features.shape[1],
                    cum * 100,
                    np.round(evr[: min(5, evr.size)], 4).tolist(),
                )
                if self.compute_pca and pca_features.shape[1] >= 3:
                    pca_xyz = pca_features[:, :3]

            # Regular UMAP/t-SNE input — usually raw keys; switches to PCA
            # features when pca_for_downstream is on (means umap_x/y/z then
            # represent PCA-then-UMAP, NOT raw-UMAP).
            features_for_reduction = (
                pca_features
                if (self.pca_for_downstream and pca_features is not None)
                else keys
            )

            # Run ALL configured reductions BEFORE writing the CSV so we can
            # bake the reduced coords as columns.
            umap_xyz = None
            pca_umap_xyz = None
            tsne_2d = None
            tsne_3d = None
            if self.compute_umap:
                with _timed(
                    f"{layer_name} | UMAP-3d ({features_for_reduction.shape[0]} rows)"
                ):
                    umap_xyz = self._cuda_umap_3d(features_for_reduction)
            if self.compute_pca_umap and pca_features is not None:
                with _timed(
                    f"{layer_name} | PCA→UMAP-3d ({pca_features.shape[0]} rows on {pca_features.shape[1]} PCs)"
                ):
                    pca_umap_xyz = self._cuda_umap_3d(pca_features)
            if self.compute_tsne_2d:
                with _timed(
                    f"{layer_name} | t-SNE-2d ({features_for_reduction.shape[0]} rows)"
                ):
                    tsne_2d = self._tsne_2d(features_for_reduction)
            if self.compute_tsne_3d:
                with _timed(
                    f"{layer_name} | t-SNE-3d ({features_for_reduction.shape[0]} rows)"
                ):
                    tsne_3d = self._tsne_3d(features_for_reduction)

            csv_path = os.path.join(out_dir, f"{layer_name}.csv")
            keys_pt_path = os.path.join(out_dir, f"{layer_name}_keys.pt")
            with _timed(f"{layer_name} | write_csv ({keys.shape[0]} rows)"):
                self._write_csv(
                    csv_path,
                    hashes,
                    embs,
                    keys,
                    umap_xyz,
                    frame_idx=frame_idx,
                    token_idx=token_idx,
                    tsne_2d=tsne_2d,
                    tsne_3d=tsne_3d,
                    pca_xyz=pca_xyz,
                    pca_umap_xyz=pca_umap_xyz,
                )
            with _timed(
                f"{layer_name} | write_keys_pt ({keys.shape[0]} x {keys.shape[1]})"
            ):
                self._write_keys_pt(keys_pt_path, keys)
            logger.info(
                "PILatentEvalVideo: wrote %s + %s (%d rows, key_dim=%d)",
                csv_path,
                os.path.basename(keys_pt_path),
                keys.shape[0],
                keys.shape[1],
            )

            if self.save_plots:
                if self.compute_umap and umap_xyz is not None:
                    with _timed(f"{layer_name} | plot UMAP-3d"):
                        self._plot_layer(
                            out_dir,
                            layer_name,
                            hashes,
                            embs,
                            umap_xyz,
                            axis_prefix="umap",
                            filename=f"{layer_name}.png",
                        )
                if self.compute_pca and pca_xyz is not None:
                    with _timed(f"{layer_name} | plot PCA-3d"):
                        self._plot_layer(
                            out_dir,
                            layer_name,
                            hashes,
                            embs,
                            pca_xyz,
                            axis_prefix="pca",
                            filename=f"{layer_name}_pca.png",
                        )
                if tsne_2d is not None:
                    with _timed(f"{layer_name} | plot t-SNE-2d"):
                        self._plot_layer(
                            out_dir,
                            layer_name,
                            hashes,
                            embs,
                            tsne_2d,
                            axis_prefix="tsne",
                            filename=f"{layer_name}_tsne2d.png",
                        )
                if tsne_3d is not None:
                    with _timed(f"{layer_name} | plot t-SNE-3d"):
                        self._plot_layer(
                            out_dir,
                            layer_name,
                            hashes,
                            embs,
                            tsne_3d,
                            axis_prefix="tsne",
                            filename=f"{layer_name}_tsne.png",
                        )

            logger.info(
                "[timing] %s | TOTAL: %.2fs",
                layer_name,
                time.perf_counter() - layer_t0,
            )

        logger.info(
            "[timing] on_validation_end TOTAL: %.2fs (%d layer slices)",
            time.perf_counter() - total_t0,
            len(self._layer_keys),
        )

    # ------------------------------------------------------------------
    # UMAP + CSV + plot
    # ------------------------------------------------------------------
    @staticmethod
    def _cuda_umap_3d(features: np.ndarray) -> np.ndarray:
        n = features.shape[0]
        if n < 4:
            return np.zeros((n, 3), dtype=np.float32)
        n_neighbors = max(2, min(15, n - 1))
        try:
            from cuml.manifold import UMAP as cuUMAP

            reducer = cuUMAP(
                n_components=3,
                n_neighbors=n_neighbors,
                metric="euclidean",
                output_type="numpy",
            )
            return reducer.fit_transform(features.astype(np.float32))
        except ImportError:
            logger.warning(
                "cuML UMAP not available — falling back to umap-learn (CPU). "
                "Install RAPIDS cuML for the CUDA path."
            )
            from umap import UMAP

            reducer = UMAP(n_components=3, n_neighbors=n_neighbors, metric="euclidean")
            return reducer.fit_transform(features.astype(np.float32))

    @staticmethod
    def _tsne_3d(features: np.ndarray) -> np.ndarray:
        n = features.shape[0]
        if n < 4:
            return np.zeros((n, 3), dtype=np.float32)
        # cuML t-SNE doesn't support n_components=3, so 3D stays on sklearn/CPU.
        perplexity = max(2, min(30, n // 3))
        perplexity = min(perplexity, n - 1)
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            random_state=0,
            metric="euclidean",
        )
        return reducer.fit_transform(features.astype(np.float32))

    @staticmethod
    def _tsne_2d(features: np.ndarray) -> np.ndarray:
        n = features.shape[0]
        if n < 4:
            return np.zeros((n, 2), dtype=np.float32)
        perplexity = max(2, min(30, n // 3))
        perplexity = min(perplexity, n - 1)
        try:
            from cuml.manifold import TSNE as cuTSNE

            return cuTSNE(
                n_components=2,
                perplexity=perplexity,
                output_type="numpy",
            ).fit_transform(features.astype(np.float32))
        except (ImportError, ValueError):
            from sklearn.manifold import TSNE

            return TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                random_state=0,
                metric="euclidean",
            ).fit_transform(features.astype(np.float32))

    @staticmethod
    def _pca(features: np.ndarray, n_components: int):
        """Run PCA on `features` (N, D). Returns (transformed (N, k),
        explained_variance_ratio (k,)). Uses cuML on GPU if available,
        sklearn on CPU otherwise. `k` is clamped to min(n_components, N, D)."""
        n, d = features.shape
        k = max(1, min(n_components, n, d))
        if n < 2:
            return np.zeros((n, k), dtype=np.float32), np.zeros((k,), dtype=np.float32)
        try:
            from cuml.decomposition import PCA as cuPCA

            reducer = cuPCA(n_components=k, output_type="numpy")
            transformed = reducer.fit_transform(features.astype(np.float32))
            ratio = np.asarray(reducer.explained_variance_ratio_, dtype=np.float32)
            return transformed, ratio
        except ImportError:
            from sklearn.decomposition import PCA as skPCA

            reducer = skPCA(n_components=k, random_state=0)
            transformed = reducer.fit_transform(features.astype(np.float32))
            return transformed.astype(
                np.float32
            ), reducer.explained_variance_ratio_.astype(np.float32)

    @staticmethod
    def _write_csv(
        path,
        hashes,
        embodiments,
        keys,
        umap_xyz,
        frame_idx=None,
        token_idx=None,
        tsne_2d=None,
        tsne_3d=None,
        pca_xyz=None,
        pca_umap_xyz=None,
    ):
        """Write per-token METADATA CSV: video_hash, embodiment, frame_idx,
        token_idx, and all cached reduction columns (umap_x/y/z, pca_umap_x/y/z,
        tsne2d_x/y, tsne3d_x/y/z, pca_x/y/z). The heavy `k0..kN` columns go
        to a sibling `<layer>_keys.pt` file via `_write_keys_pt`."""
        has_umap = umap_xyz is not None
        has_pca_umap = pca_umap_xyz is not None
        has_tsne2d = tsne_2d is not None
        has_tsne3d = tsne_3d is not None
        has_pca = pca_xyz is not None
        has_indices = frame_idx is not None and token_idx is not None
        header = ["video_hash", "embodiment"]
        if has_indices:
            header += ["frame_idx", "token_idx"]
        if has_umap:
            header += ["umap_x", "umap_y", "umap_z"]
        if has_pca_umap:
            header += ["pca_umap_x", "pca_umap_y", "pca_umap_z"]
        if has_tsne2d:
            header += ["tsne2d_x", "tsne2d_y"]
        if has_tsne3d:
            header += ["tsne3d_x", "tsne3d_y", "tsne3d_z"]
        if has_pca:
            header += ["pca_x", "pca_y", "pca_z"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(keys.shape[0]):
                row = [hashes[i], embodiments[i]]
                if has_indices:
                    row += [int(frame_idx[i]), int(token_idx[i])]
                if has_umap:
                    row += [
                        float(umap_xyz[i, 0]),
                        float(umap_xyz[i, 1]),
                        float(umap_xyz[i, 2]),
                    ]
                if has_pca_umap:
                    row += [
                        float(pca_umap_xyz[i, 0]),
                        float(pca_umap_xyz[i, 1]),
                        float(pca_umap_xyz[i, 2]),
                    ]
                if has_tsne2d:
                    row += [float(tsne_2d[i, 0]), float(tsne_2d[i, 1])]
                if has_tsne3d:
                    row += [
                        float(tsne_3d[i, 0]),
                        float(tsne_3d[i, 1]),
                        float(tsne_3d[i, 2]),
                    ]
                if has_pca:
                    row += [
                        float(pca_xyz[i, 0]),
                        float(pca_xyz[i, 1]),
                        float(pca_xyz[i, 2]),
                    ]
                writer.writerow(row)

    @staticmethod
    def _write_keys_pt(path: str, keys: np.ndarray, dtype: str = "float32"):
        """Save the (N, D) raw-key matrix to a torch tensor file. Default
        dtype keeps eval-time precision (float32); set to 'float16' to
        halve disk usage with negligible loss for visualization."""
        if dtype == "float16":
            tensor = torch.from_numpy(keys.astype(np.float16, copy=False))
        elif dtype == "bfloat16":
            tensor = torch.from_numpy(keys.astype(np.float32, copy=False)).to(
                torch.bfloat16
            )
        else:
            tensor = torch.from_numpy(keys.astype(np.float32, copy=False))
        torch.save(tensor.contiguous(), path)

    # ------------------------------------------------------------------
    # Replot from existing CSVs (skip the model forward entirely)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_csv_for_replot(path):
        """Read CSV (metadata + cached reductions) and load keys from the
        sibling `<layer>_keys.pt` torch file when present. Falls back to
        in-CSV `k0..kN` columns for legacy CSVs that have keys inline.
        Returns dict with: hashes, embs, frame_idx, token_idx, keys (N, D),
        umap_xyz, tsne_2d, tsne_3d, pca_xyz."""
        rows_hash, rows_emb, rows_frame, rows_token = [], [], [], []
        rows_umap, rows_tsne2d, rows_tsne3d, rows_pca = [], [], [], []
        rows_keys_inline = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_umap = {"umap_x", "umap_y", "umap_z"}.issubset(fieldnames)
            has_tsne2d = {"tsne2d_x", "tsne2d_y"}.issubset(fieldnames)
            has_tsne3d = {"tsne3d_x", "tsne3d_y", "tsne3d_z"}.issubset(fieldnames)
            has_pca = {"pca_x", "pca_y", "pca_z"}.issubset(fieldnames)
            has_frame = "frame_idx" in fieldnames
            has_token = "token_idx" in fieldnames
            k_cols = sorted(
                [c for c in fieldnames if c.startswith("k") and c[1:].isdigit()],
                key=lambda c: int(c[1:]),
            )
            has_inline_keys = bool(k_cols)
            for r in reader:
                rows_hash.append(r["video_hash"])
                rows_emb.append(r.get("embodiment", ""))
                rows_frame.append(int(r["frame_idx"]) if has_frame else -1)
                rows_token.append(int(r["token_idx"]) if has_token else -1)
                if has_umap:
                    rows_umap.append(
                        (float(r["umap_x"]), float(r["umap_y"]), float(r["umap_z"]))
                    )
                if has_tsne2d:
                    rows_tsne2d.append((float(r["tsne2d_x"]), float(r["tsne2d_y"])))
                if has_tsne3d:
                    rows_tsne3d.append(
                        (
                            float(r["tsne3d_x"]),
                            float(r["tsne3d_y"]),
                            float(r["tsne3d_z"]),
                        )
                    )
                if has_pca:
                    rows_pca.append(
                        (float(r["pca_x"]), float(r["pca_y"]), float(r["pca_z"]))
                    )
                if has_inline_keys:
                    rows_keys_inline.append([float(r[c]) for c in k_cols])

        # Resolve `keys`: prefer sibling .pt file (new format); fall back to
        # in-CSV inline columns (legacy format).
        keys_pt_path = (
            path[:-4] + "_keys.pt" if path.endswith(".csv") else path + "_keys.pt"
        )
        if os.path.isfile(keys_pt_path):
            try:
                tensor = torch.load(keys_pt_path, map_location="cpu", weights_only=True)
            except Exception:
                tensor = torch.load(keys_pt_path, map_location="cpu")
            keys = tensor.to(torch.float32).cpu().numpy()
        elif has_inline_keys:
            keys = np.asarray(rows_keys_inline, dtype=np.float32)
        else:
            # No keys anywhere — meanpool / recompute paths will need to skip.
            keys = np.empty((len(rows_hash), 0), dtype=np.float32)

        return {
            "hashes": rows_hash,
            "embs": rows_emb,
            "frame_idx": rows_frame if has_frame else None,
            "token_idx": rows_token if has_token else None,
            "keys": keys,
            "umap_xyz": np.asarray(rows_umap, dtype=np.float32) if has_umap else None,
            "tsne_2d": np.asarray(rows_tsne2d, dtype=np.float32)
            if has_tsne2d
            else None,
            "tsne_3d": np.asarray(rows_tsne3d, dtype=np.float32)
            if has_tsne3d
            else None,
            "pca_xyz": np.asarray(rows_pca, dtype=np.float32) if has_pca else None,
        }

    def run(self, trainer, model, datamodule, cfg):
        """Latent-eval routing: if a prior run already wrote per-layer
        CSVs, reuse them and skip the model forward. Otherwise do
        normal load-ckpt + validate, honoring ``pretrained=true``."""
        import os
        import re

        import torch

        from egomimic.utils.hydra_resolvers import (
            model_time_from_ckpt,
            model_type_from_ckpt,
        )

        log = __import__("logging").getLogger(__name__)

        # --- check for existing CSVs ---
        existing = None
        if str(cfg.get("mode")) == "eval" and not cfg.get("force_reeval", False):
            name = cfg.get("name")
            description = cfg.get("description")
            ckpt_path = cfg.get("ckpt_path")
            mtype = model_type_from_ckpt(ckpt_path)
            mtime = model_time_from_ckpt(ckpt_path)
            parent = os.path.join(
                "logs", str(name), "latent_eval", str(mtype), str(mtime)
            )
            if os.path.isdir(parent):
                pat = re.compile(
                    rf"^{re.escape(str(description))}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}$"
                )
                candidates = []
                for rd in sorted(os.listdir(parent)):
                    if not pat.match(rd):
                        continue
                    epoch_root = os.path.join(parent, rd, "latents")
                    if not os.path.isdir(epoch_root):
                        continue
                    for ed in sorted(os.listdir(epoch_root)):
                        full = os.path.join(epoch_root, ed)
                        if os.path.isdir(full) and any(
                            f.endswith(".csv") for f in os.listdir(full)
                        ):
                            candidates.append(full)
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p))
                    existing = candidates[-1]

        if existing is not None:
            log.info(
                f"Reusing existing CSVs from {existing} — skipping model forward. "
                f"To force re-eval, pass `+force_reeval=true` on the CLI."
            )
            new_dir = os.path.join(self.latent_dir(), "epoch_0")
            self.rebuild_from_csvs(existing, out_dir=new_dir)
            return

        # --- normal eval: load ckpt + validate ---
        ckpt_path = cfg.get("ckpt_path")
        pretrained = bool(cfg.get("pretrained", False))
        if ckpt_path and not pretrained:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            log.info(f"Loaded weights from {ckpt_path}")
        elif ckpt_path and pretrained:
            log.info(
                f"pretrained=true → skipping checkpoint load. "
                f"Routing output under {ckpt_path}'s folder for side-by-side comparison."
            )
        log.info("Starting evaluation!")
        trainer.validate(model=model, datamodule=datamodule)

    def rebuild_from_csvs(self, source_dir: str, out_dir: str | None = None):
        """Re-render plots from an existing per-layer CSV directory using the
        currently-configured flags. CSVs from `source_dir` are symlinked into
        `out_dir`; nothing is recomputed for reductions whose columns are
        already cached. If a reduction is requested but its columns are
        missing, this method will compute it for plotting only — but it will
        NOT write the new column back to the CSV. To persist new columns
        into existing CSVs, use the standalone
        `egomimic/scripts/data_visualization/add_missing_reductions.py`
        script instead.
        """
        out_dir = out_dir or os.path.join(self.latent_dir(), "epoch_0")
        os.makedirs(out_dir, exist_ok=True)
        csvs = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
        csvs = [p for p in csvs if os.path.basename(p) != "comparison.csv"]
        if not csvs:
            logger.warning(
                "rebuild_from_csvs: no per-layer CSVs found in %s", source_dir
            )
            return

        logger.info(
            "rebuild_from_csvs: rendering plots from %d CSVs in %s -> %s",
            len(csvs),
            source_dir,
            out_dir,
        )
        total_t0 = time.perf_counter()
        for csv_path in csvs:
            layer_name = os.path.splitext(os.path.basename(csv_path))[0]
            with _timed(f"{layer_name} | read_csv"):
                d = self._read_csv_for_replot(csv_path)
            hashes, embs = d["hashes"], d["embs"]
            keys = d["keys"]
            n = len(hashes)
            if n == 0:
                logger.warning("rebuild_from_csvs: %s is empty; skipping", layer_name)
                continue

            # Make the source CSV (and sibling _keys.pt, if any) available
            # in the new run dir via symlink.
            src_pt = os.path.join(source_dir, f"{layer_name}_keys.pt")
            new_pt = os.path.join(out_dir, f"{layer_name}_keys.pt")
            if os.path.isfile(src_pt) and not os.path.exists(new_pt):
                try:
                    os.symlink(os.path.abspath(src_pt), new_pt)
                except (OSError, NotImplementedError):
                    shutil.copy2(src_pt, new_pt)
            new_csv = os.path.join(out_dir, f"{layer_name}.csv")
            if not os.path.exists(new_csv):
                try:
                    os.symlink(os.path.abspath(csv_path), new_csv)
                except (OSError, NotImplementedError):
                    shutil.copy2(csv_path, new_csv)

            # Pull cached coords; compute on-the-fly only if missing AND
            # requested. Computed coords are NOT written back to the CSV.
            features_for_reduction = keys
            pca_xyz = d.get("pca_xyz")
            if self.compute_pca and pca_xyz is None:
                with _timed(
                    f"{layer_name} | PCA-{self.pca_n_components} (transient, {n} rows)"
                ):
                    pca_features, evr = self._pca(keys, self.pca_n_components)
                cum = float(np.cumsum(evr)[-1]) if evr.size else 0.0
                logger.info(
                    "[PCA] %s: %d components -> cumulative explained variance = %.2f%%",
                    layer_name,
                    pca_features.shape[1],
                    cum * 100,
                )
                if pca_features.shape[1] >= 3:
                    pca_xyz = pca_features[:, :3]
                if self.pca_for_downstream:
                    features_for_reduction = pca_features

            umap_xyz = d.get("umap_xyz")
            if self.compute_umap and umap_xyz is None:
                with _timed(f"{layer_name} | UMAP-3d (transient, {n} rows)"):
                    umap_xyz = self._cuda_umap_3d(features_for_reduction)

            if not self.save_plots:
                continue

            if self.compute_umap and umap_xyz is not None:
                with _timed(f"{layer_name} | plot UMAP-3d"):
                    self._plot_layer(
                        out_dir,
                        layer_name,
                        hashes,
                        embs,
                        umap_xyz,
                        axis_prefix="umap",
                        filename=f"{layer_name}.png",
                    )
            if self.compute_pca and pca_xyz is not None:
                with _timed(f"{layer_name} | plot PCA-3d"):
                    self._plot_layer(
                        out_dir,
                        layer_name,
                        hashes,
                        embs,
                        pca_xyz,
                        axis_prefix="pca",
                        filename=f"{layer_name}_pca.png",
                    )
            if self.compute_tsne_2d:
                with _timed(
                    f"{layer_name} | t-SNE-2d ({features_for_reduction.shape[0]} rows)"
                ):
                    tsne_2d = self._tsne_2d(features_for_reduction)
                with _timed(f"{layer_name} | plot t-SNE-2d"):
                    self._plot_layer(
                        out_dir,
                        layer_name,
                        hashes,
                        embs,
                        tsne_2d,
                        axis_prefix="tsne",
                        filename=f"{layer_name}_tsne2d.png",
                    )
            if self.compute_tsne_3d:
                with _timed(
                    f"{layer_name} | t-SNE-3d ({features_for_reduction.shape[0]} rows)"
                ):
                    tsne_xyz = self._tsne_3d(features_for_reduction)
                with _timed(f"{layer_name} | plot t-SNE-3d"):
                    self._plot_layer(
                        out_dir,
                        layer_name,
                        hashes,
                        embs,
                        tsne_xyz,
                        axis_prefix="tsne",
                        filename=f"{layer_name}_tsne.png",
                    )

        logger.info(
            "[timing] rebuild_from_csvs TOTAL: %.2fs",
            time.perf_counter() - total_t0,
        )

    def _plot_layer(
        self,
        out_dir,
        layer_name,
        hashes,
        embodiments,
        coords,
        *,
        axis_prefix,
        filename,
    ):
        """
        Render a 2D or 3D scatter of the layer's reduced coordinates (dim is
        inferred from `coords.shape[1]`), one dot per row, colored by hash.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            logger.warning("matplotlib not available; skipping plot for %s", layer_name)
            return

        d = coords.shape[1]
        # Choose grouping: by embodiment (aria/eva) or by individual hash.
        groups = embodiments if self.color_by == "embodiment" else hashes
        groups_arr = np.array(groups)
        unique_groups = sorted(set(groups))
        cmap = plt.get_cmap("tab10" if len(unique_groups) <= 10 else "tab20")
        color_for = {g: cmap(i % cmap.N) for i, g in enumerate(unique_groups)}

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d") if d == 3 else fig.add_subplot(111)
        for g in unique_groups:
            mask = groups_arr == g
            pts = coords[mask]
            label = f"{g} (n={int(mask.sum())})"
            if d == 3:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    c=[color_for[g]],
                    s=8,
                    alpha=0.5,
                    label=label,
                )
            else:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c=[color_for[g]],
                    s=8,
                    alpha=0.5,
                    label=label,
                )

        ax.set_title(f"{layer_name} ({axis_prefix} {d}D)")
        ax.set_xlabel(f"{axis_prefix}_x")
        ax.set_ylabel(f"{axis_prefix}_y")
        if d == 3:
            ax.set_zlabel(f"{axis_prefix}_z")
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename), dpi=140)
        plt.close(fig)
