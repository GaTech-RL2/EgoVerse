from collections import OrderedDict

import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class Algo:
    """
    Base class for all algorithms.  These functions are expected by Pytorch Lightning trainer (pl_model.py)
    """

    def __init__(self):
        pass

    # ----- Shared embodiment-key resolution / obs assembly / log defaults -----
    #
    # These three helpers were byte-identical across HNet (packed_base.py),
    # DFoT (diffusion/algo.py), and WindowedBC (bc.py inherits via HNet).
    # They are hoisted here (collapse c5) so the single definition is shared
    # by every policy subclass; behaviour is preserved by construction since
    # the moved text is identical. Subclasses call ``_resolve_embodiment_keys``
    # from their ``__init__`` and inherit ``_build_obs`` / ``log_info``.

    def _resolve_embodiment_keys(self, norm_stats):
        """Resolve per-embodiment keys via norm_stats (which owns the
        MultiDataset key topology — same surface HPT uses).

        Requires ``self.domains`` and ``self.ac_keys`` to be set. Populates
        ``self.embodiment_ids`` and the per-embodiment-id ``proprio_keys`` /
        ``lang_keys`` / ``camera_keys`` / ``resolved_ac_keys`` dicts.
        """
        self.embodiment_ids = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for emb in self.domains:
            emb_id = get_embodiment_id(emb)
            self.embodiment_ids[emb] = emb_id
            self.proprio_keys[emb_id] = []
            self.lang_keys[emb_id] = []
            self.camera_keys[emb_id] = []
            for key in norm_stats.keys_of_type("action_keys", emb_id):
                if (
                    norm_stats.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in norm_stats.keys_of_type("proprio_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("camera_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)

    def _build_obs(self, _batch, emb_id):
        obs = {}
        for key in (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        ):
            if key in _batch:
                obs[key] = _batch[key]
        return obs

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            batch (dict) : processed dict of batches of form
            <embodiment_id> : {<dataset_keys> torch.Tensor}
        """
        raise NotImplementedError(
            "Must implement process_batch_for_training in subclass"
        )

    def forward_training(self, batch):
        """
        One iteration of training.  Compute forward pass and compute losses.  Return predictions dictionary.  ACT also calculates loss here.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D), loss_key_name: torch.Tensor (1)}
        """
        raise NotImplementedError("Must implement forward_training in subclass")

    def forward_eval(self, batch):
        """
        Compute forward pass and return network outputs in @predictions dict.
        This is the only eval-time entry point the model is expected to expose;
        all metric computation and visualization is performed by Eval classes
        on top of these predictions.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            predictions (dict): {ac_key: torch.Tensor (B, Seq, D)}
        """
        raise NotImplementedError("Must implement forward_eval in subclass")

    def compute_losses(self, predictions, batch):
        """
        Compute losses based on network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training (see docstring for expected keys/shapes)
        Returns:
            losses (dict): dictionary of losses computed over the batch
                loss_key_name: torch.Tensor (1)

        Default reducer (dedup collapse c7): sum each embodiment's
        ``{emb_id}_action_loss`` prediction into the aggregate ``action_loss``
        (mean over present embodiments), exposing each per-embodiment term as
        ``emb{emb_id}_action_loss``. This is the pure skeleton shared by the
        DFoT algo; subclasses whose loss carries *extra* terms (VAE
        recon/kl/lpips, HNet ratio_loss, HPT/PI domain-count division) still
        override with their own superset reducer.
        """
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for emb_id in batch.keys():
            a = predictions[f"{emb_id}_action_loss"]
            loss_dict[f"emb{emb_id}_action_loss"] = a
            total = total + a
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of losses returned by compute_losses
                losses:
                    loss_key_name: torch.Tensor (1)
        Returns:
            loss_log (dict): name -> summary statistic

        Default implementation (shared by HNet/DFoT/WindowedBC, collapse c5):
        surfaces ``action_loss`` as ``Loss`` plus every loss under its own
        key. Subclasses may still override for richer logging.
        """
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log
