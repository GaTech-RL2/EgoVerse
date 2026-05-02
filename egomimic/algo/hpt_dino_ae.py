"""Two-stage HPT-DINO action-expert algorithm.

Stage 1 (world model, with trunk):
  inputs   = DINO embedding at BOS + sampled language prompt
  target   = DINO embedding at EOS  (flow-matched, action_horizon=1)

Stage 2 (action expert, no trunk):
  inputs   = GT EOS DINO (training) / predicted EOS DINO (inference)
             + obs_t (front image, wrist images on EVA, proprio)
  target   = action chunk (flow-matched, action_horizon=100)

Training is staged, not joint. The active stage is determined by global step:
  step <  stage1_steps           -> train Stage 1, Stage 2 frozen
  step >= stage1_steps           -> train Stage 2, Stage 1 frozen

Each stage's pretrained weights can be loaded independently from disk so the
two stages can be trained in separate runs (e.g. set ``stage1_steps=0`` and
load ``stage1_pretrained_checkpoint`` to train only Stage 2).
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
from overrides import override
from torchmetrics import MeanSquaredError

from egomimic.algo.algo import Algo
from egomimic.algo.hpt import HPTModel
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.utils.egomimicUtils import get_sinusoid_encoding_table

_PASSTHROUGH_TYPES = {"camera_keys", "proprio_keys", "lang_keys", "annotation_keys"}


class ActionExpertModel(nn.Module):
    """Stage-2 module: stems + encoders + flow head, no trunk.

    Concatenates per-modality stem latent tokens into a single conditioning
    sequence and feeds it to a flow-matching head that predicts actions.
    """

    def __init__(
        self,
        domains: list[str],
        stem_specs: dict | None = None,
        shared_stem_specs: dict | None = None,
        shared_obs_keys: list[str] | None = None,
        encoder_specs: dict | None = None,
        head_spec: Any = None,
    ):
        super().__init__()
        self.domains = list(domains)
        self.shared_keys = list(shared_obs_keys or [])

        self.stems = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.modalities: dict[str, list[str]] = {}

        for modality, spec in (shared_stem_specs or {}).items():
            self._register_stem("shared", modality, spec)

        for domain in self.domains:
            self.modalities.setdefault(domain, [])
            for modality, spec in ((stem_specs or {}).get(domain) or {}).items():
                self._register_stem(domain, modality, spec)

        for modality, encoder in (encoder_specs or {}).items():
            self.encoders[modality] = encoder

        self.head = head_spec
        self.apply(_init_weights)

    def _register_stem(self, domain: str, modality: str, spec: nn.Module):
        if domain != "shared":
            self.modalities.setdefault(domain, []).append(modality)
        stem_name = f"{domain}_{modality}"
        self.stems[stem_name] = spec
        if hasattr(spec, "init_cross_attn"):
            spec.init_cross_attn(spec.specs.cross_attn)

    def encode(self, domain: str, data: dict) -> torch.Tensor:
        feats = []
        modalities = list(self.modalities.get(domain, [])) + list(self.shared_keys)
        for modality in modalities:
            if modality not in data:
                continue
            stem_domain = "shared" if modality in self.shared_keys else domain
            stem = self.stems[f"{stem_domain}_{modality}"]

            if modality in self.encoders:
                data[modality] = self.encoders[modality](data[modality])

            x = data[modality]
            data_shape = x.shape
            position = get_sinusoid_encoding_table(
                0, data_shape[1] * int(np.prod(data_shape[2:-1])), data_shape[-1]
            ).to(x)
            position = einops.repeat(
                position, "b h w -> (repeat b) h w", repeat=data_shape[0]
            )
            x = x + position.view(x.shape)
            feats.append(stem.compute_latent(x))
        if not feats:
            raise RuntimeError(f"No stage-2 modalities present for domain {domain}")
        return torch.cat(feats, dim=-2)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        domain, data = batch["domain"], batch["data"]
        cond = self.encode(domain, data)
        return self.head.compute_loss(cond, data)

    def forward(self, domain: str, data: dict) -> torch.Tensor:
        cond = self.encode(domain, data)
        return self.head((cond, domain))


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class HPTDinoAEPolicy(nn.Module):
    """Composite module holding Stage-1 (HPTModel + trunk) and Stage-2 (ActionExpertModel)."""

    def __init__(self, stage1: HPTModel, stage2: ActionExpertModel):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2


class HPTDinoAE(Algo):
    """Two-stage HPT-DINO algorithm. See module docstring."""

    def __init__(
        self,
        data_schematic,
        camera_transforms,
        train_image_augs,
        eval_image_augs,
        # ---------- Stage 1 (world model with trunk) ----------
        stage1_trunk: dict,
        stage1_shared_stem_specs: dict,
        stage1_shared_obs_keys: list[str],
        stage1_head_specs: dict,
        # ---------- Stage 2 (action expert, no trunk) ----------
        stage2_shared_stem_specs: dict,
        stage2_shared_obs_keys: list[str],
        stage2_stem_specs: dict,
        stage2_head_spec: Any,
        # ---------- Encoders (shared module dicts) ----------
        stage1_encoder_specs: dict | None = None,
        stage2_encoder_specs: dict | None = None,
        # ---------- Domains and target keys ----------
        domains: list[str] = None,
        ac_keys: dict | None = None,
        shared_ac_key_stage1: str = "dino_front_T",
        shared_ac_key_stage2: str = "actions_cartesian",
        # ---------- Training schedule ----------
        stage1_steps: int = 0,
        stage2_steps: int = 0,
        stage1_pretrained_checkpoint: str | None = None,
        stage2_pretrained_checkpoint: str | None = None,
        # ---------- Annotation sampling ----------
        # During training the sampled_prompt is drawn uniformly at random from
        # the per-frame annotation list; during eval the index here is used so
        # validation runs are deterministic across epochs. Out-of-range indices
        # are clamped to the last available annotation per sample.
        eval_annotation_index: int = 0,
        default_prompt: str = "",
        # ---------- Misc ----------
        viz_func=None,
        **kwargs,
    ):
        self.data_schematic = data_schematic
        self.viz_func = viz_func
        self.camera_transforms = camera_transforms
        self.train_image_augs = train_image_augs
        self.eval_image_augs = eval_image_augs

        self.domains = list(domains)
        self.ac_keys = dict(ac_keys or {})
        self.shared_ac_key_stage1 = shared_ac_key_stage1
        self.shared_ac_key_stage2 = shared_ac_key_stage2

        self.stage1_steps = int(stage1_steps)
        self.stage2_steps = int(stage2_steps)
        self.training_step = 0
        self._stage1_frozen = False
        self._stage2_frozen = False

        self.eval_annotation_index = int(eval_annotation_index)
        self.default_prompt = default_prompt

        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---- Build Stage 1 (HPTModel reused as-is) ----
        # Domain is always "shared" for Stage 1; we don't use the
        # shared_keys/per-domain split that HPT cotrain uses.
        stage1_model = HPTModel(**stage1_trunk)
        stage1_model.diffusion = True
        stage1_model.device = self.device
        stage1_model.shared_keys = []
        stage1_model.auxiliary_ac_keys = {}
        stage1_model.shared_action = False

        stage1_model.init_domain_stem("shared", stage1_shared_stem_specs)
        stage1_model.init_domain_head("shared", stage1_head_specs["shared"])
        for modality, encoder in (stage1_encoder_specs or {}).items():
            stage1_model.init_encoder(modality, encoder)
        stage1_model.finalize_modules()

        # ---- Build Stage 2 (custom no-trunk module) ----
        stage2_model = ActionExpertModel(
            domains=self.domains,
            stem_specs=stage2_stem_specs,
            shared_stem_specs=stage2_shared_stem_specs,
            shared_obs_keys=stage2_shared_obs_keys,
            encoder_specs=stage2_encoder_specs,
            head_spec=stage2_head_spec,
        )

        self.nets = nn.ModuleDict()
        self.nets["policy"] = HPTDinoAEPolicy(stage1_model, stage2_model).to(
            self.device
        )
        self.nets = self.nets.float()

        # ---- Resolve schematic-driven key lists per embodiment ----
        self.camera_keys: dict[int, list[str]] = {}
        self.proprio_keys: dict[int, list[str]] = {}
        self.lang_keys: dict[int, list[str]] = {}
        self.annotation_keys: dict[int, list[str]] = {}
        self.action_keys: dict[int, list[str]] = {}
        for domain in self.domains:
            eid = get_embodiment_id(domain)
            self.camera_keys[eid] = list(
                data_schematic.keys_of_type("camera_keys", eid)
            )
            self.proprio_keys[eid] = list(
                data_schematic.keys_of_type("proprio_keys", eid)
            )
            self.lang_keys[eid] = list(data_schematic.keys_of_type("lang_keys", eid))
            self.annotation_keys[eid] = list(
                data_schematic.keys_of_type("annotation_keys", eid)
            )
            self.action_keys[eid] = list(
                data_schematic.keys_of_type("action_keys", eid)
            )

        # ---- Optional pretrained loading per stage ----
        if stage1_pretrained_checkpoint:
            self.load_stage1_pretrained(stage1_pretrained_checkpoint)
        if stage2_pretrained_checkpoint:
            self.load_stage2_pretrained(stage2_pretrained_checkpoint)

        self._set_active_stage(self.current_stage())

    # --------------------------------------------------------------
    # Stage scheduling and freezing
    # --------------------------------------------------------------

    def current_stage(self) -> int:
        return 1 if self.training_step < self.stage1_steps else 2

    def _set_active_stage(self, stage: int) -> None:
        s1_train = stage == 1
        s2_train = stage == 2
        for p in self.nets["policy"].stage1.parameters():
            p.requires_grad = s1_train
        for p in self.nets["policy"].stage2.parameters():
            p.requires_grad = s2_train
        self._stage1_frozen = not s1_train
        self._stage2_frozen = not s2_train

    # --------------------------------------------------------------
    # Pretrained loading
    # --------------------------------------------------------------

    @staticmethod
    def _extract_stage_state_dict(path: str, stage_name: str) -> dict:
        """Pull a stage's submodule state_dict from a checkpoint, tolerant to
        the various key prefixes that can appear in disk checkpoints
        (``state_dict`` wrapper, ``model.``, ``nets.policy.``, etc.).
        """
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        marker = f"{stage_name}."
        out: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if marker not in k:
                continue
            # Take everything *after* the stage marker so we always end up with
            # keys rooted at the stage submodule (e.g. ``trunk.trunk.blocks.0...``).
            out[k.split(marker, 1)[1]] = v
        return out

    def load_stage1_pretrained(self, path: str) -> None:
        sd = self._extract_stage_state_dict(path, "stage1")
        missing, unexpected = self.nets["policy"].stage1.load_state_dict(
            sd, strict=False
        )
        print(
            f"[HPTDinoAE] loaded stage1 from {path}: "
            f"{len(sd)} matched, {len(missing)} missing, {len(unexpected)} unexpected"
        )

    def load_stage2_pretrained(self, path: str) -> None:
        sd = self._extract_stage_state_dict(path, "stage2")
        missing, unexpected = self.nets["policy"].stage2.load_state_dict(
            sd, strict=False
        )
        print(
            f"[HPTDinoAE] loaded stage2 from {path}: "
            f"{len(sd)} matched, {len(missing)} missing, {len(unexpected)} unexpected"
        )

    # --------------------------------------------------------------
    # Batch preparation
    # --------------------------------------------------------------

    def _sample_annotation(self, annotations: list) -> str:
        """Pick one annotation string out of a per-sample list.

        Training (``self.nets.training``): uniform-random. Eval: take index
        ``self.eval_annotation_index``, clamped to the last available entry.
        Empty lists fall back to ``self.default_prompt``.
        """
        if not annotations:
            return self.default_prompt
        if self.nets.training:
            return annotations[random.randint(0, len(annotations) - 1)]
        idx = min(self.eval_annotation_index, len(annotations) - 1)
        return annotations[idx]

    def _sample_prompts(self, batch_lang_lists) -> list:
        """list[list[str]] -> list[str]; one annotation per sample."""
        return [self._sample_annotation(sample) for sample in batch_lang_lists]

    @override
    def process_batch_for_training(self, batch):
        """Per-embodiment batch prep. Mode-aware: in training the language
        prompt is sampled uniformly from each sample's annotation list, in
        eval mode it's taken at ``eval_annotation_index`` so validation is
        deterministic.
        """
        lang_keys_by_eid = self.lang_keys
        processed = {}
        for embodiment_name, _batch in batch.items():
            eid = get_embodiment_id(embodiment_name)
            processed[eid] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, eid)
                if key_name is None:
                    continue
                processed[eid][key_name] = value

            # Resolve language prompts: list[list[str]] -> list[str].
            for lkey in lang_keys_by_eid.get(eid, []):
                value = processed[eid].get(lkey)
                if isinstance(value, list) and value and isinstance(value[0], list):
                    processed[eid][lkey] = self._sample_prompts(value)

            processed[eid] = self.data_schematic.normalize_data(processed[eid], eid)
            processed[eid]["embodiment"] = torch.tensor(
                [eid], device=self.device, dtype=torch.int64
            )
            for key, value in processed[eid].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed[eid][key] = value
        return processed

    # --------------------------------------------------------------
    # Stage-specific data routing
    # --------------------------------------------------------------

    def _build_stage1_data(self, eid: int, batch: dict) -> dict:
        data: dict = {}
        # DINO BOS goes through proprio path -> "state_dino_front_1" carrying (B, 1, 1, D).
        bos_key = "dino_front_1"
        if bos_key in batch:
            data[f"state_{bos_key}"] = batch[bos_key].unsqueeze(1)

        # Sampled prompt comes from the t-suffixed annotation list.
        prompt_key = self._find_first(batch, ["sampled_prompt", "annotations_t"])
        if prompt_key is not None:
            data["sampled_prompt"] = batch[prompt_key]

        # EOS DINO is the prediction target.
        target_key = self.shared_ac_key_stage1
        if target_key not in batch:
            raise KeyError(
                f"Stage 1 target '{target_key}' missing for embodiment {eid}"
            )
        target = batch[target_key]
        if target.ndim == 2:
            target = target.unsqueeze(1)  # (B, 1, D)
        data["action"] = target
        data["embodiment"] = batch["embodiment"]
        data["pad_mask"] = torch.ones(
            target.shape[0], target.shape[1], 1, device=target.device
        )
        return data

    def _build_stage2_data(
        self,
        eid: int,
        batch: dict,
        eos_dino_override: torch.Tensor | None = None,
    ) -> dict:
        """Build the data dict consumed by Stage 2.

        ``eos_dino_override`` lets the combined-inference path inject the
        Stage-1 prediction in place of the GT EOS DINO embedding.
        """
        data: dict = {}

        # EOS DINO routed as proprio token (GT during training, predicted at
        # combined inference time).
        eos_key = "dino_front_T"
        eos_value = (
            eos_dino_override if eos_dino_override is not None else batch.get(eos_key)
        )
        if eos_value is not None:
            if eos_value.ndim == 3 and eos_value.shape[1] == 1:
                eos_value = eos_value.squeeze(1)
            data[f"state_{eos_key}"] = eos_value.unsqueeze(1)

        # Current-frame images.
        for key in ("front_img_t", "right_wrist_img_t", "left_wrist_img_t"):
            if key in batch:
                tensor = batch[key]
                if self.nets.training and key in self._stage2_encoders():
                    tensor = self.train_image_augs(tensor)
                elif self.eval_image_augs and key in self._stage2_encoders():
                    tensor = self.eval_image_augs(tensor)
                data[key] = tensor.unsqueeze(1).unsqueeze(1)

        # Current-frame proprio (e.g. concat'd ``observations.state.ee_pose_t``).
        for key in self.proprio_keys[eid]:
            if not key.endswith("_t") or key == "dino_front_t":
                continue
            if key in batch:
                data[f"state_{key}"] = batch[key].unsqueeze(1)

        ac_key = self.shared_ac_key_stage2
        if ac_key not in batch:
            raise KeyError(f"Stage 2 target '{ac_key}' missing for embodiment {eid}")
        data["action"] = batch[ac_key]
        data["embodiment"] = batch["embodiment"]
        data["pad_mask"] = torch.ones(
            data["action"].shape[0], data["action"].shape[1], 1, device=self.device
        )
        return data

    def _stage2_encoders(self) -> set:
        return set(self.nets["policy"].stage2.encoders.keys())

    @staticmethod
    def _find_first(batch: dict, candidates):
        for c in candidates:
            if c in batch:
                return c
        return None

    # --------------------------------------------------------------
    # Forward / loss
    # --------------------------------------------------------------

    @override
    def forward_training(self, batch):
        self.training_step += 1
        stage = self.current_stage()
        if (stage == 1 and self._stage1_frozen) or (stage == 2 and self._stage2_frozen):
            self._set_active_stage(stage)

        predictions = OrderedDict()
        predictions["stage"] = stage
        for eid, _batch in batch.items():
            embodiment_name = get_embodiment(eid).lower()
            if stage == 1:
                data = self._build_stage1_data(eid, _batch)
                hpt_batch = {"domain": "shared", "data": data}
                loss = self.nets["policy"].stage1.compute_loss(hpt_batch)
                predictions[f"{embodiment_name}_stage1_loss"] = loss
            else:
                data = self._build_stage2_data(eid, _batch)
                ae_batch = {"domain": embodiment_name, "data": data}
                loss = self.nets["policy"].stage2.compute_loss(ae_batch)
                predictions[f"{embodiment_name}_stage2_loss"] = loss
        return predictions

    @override
    def compute_losses(self, predictions, batch):
        loss_dict = OrderedDict()
        total = torch.tensor(0.0, device=self.device)
        stage = predictions.get("stage", self.current_stage())
        for eid in batch:
            name = get_embodiment(eid).lower()
            key = f"{name}_stage{stage}_loss"
            if key in predictions:
                loss_dict[key] = predictions[key]
                total = total + predictions[key]
        loss_dict["action_loss"] = total / max(1, len(batch))
        loss_dict[f"stage{stage}_active"] = torch.tensor(1.0, device=self.device)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for key, value in info["losses"].items():
            log[key] = value.item() if isinstance(value, torch.Tensor) else value
        return log

    # --------------------------------------------------------------
    # Eval
    # --------------------------------------------------------------

    @override
    def forward_eval(self, batch):
        """Active-stage eval forward. Stage 1 returns predicted EOS DINO;
        Stage 2 returns predicted actions conditioned on GT EOS DINO. The
        combined Stage1->Stage2 inference path lives in the eval module.
        """
        unnorm_preds: dict[str, torch.Tensor] = {}
        stage = self.current_stage()
        for eid, _batch in batch.items():
            embodiment_name = get_embodiment(eid).lower()
            if stage == 1:
                data = self._build_stage1_data(eid, _batch)
                pred = self.nets["policy"].stage1.forward("shared", data)
                pred_tensor = pred.get("shared", next(iter(pred.values())))
                T = data["action"].shape[1]
                D = data["action"].shape[-1]
                preds = {self.shared_ac_key_stage1: pred_tensor[:, :T, :D]}
            else:
                data = self._build_stage2_data(eid, _batch)
                pred_tensor = self.nets["policy"].stage2.forward(embodiment_name, data)
                ref = data["action"]
                T, D = ref.shape[1], ref.shape[-1]
                preds = {self.shared_ac_key_stage2: pred_tensor[:, :T, :D]}

            unnorm = self.data_schematic.unnormalize_data(preds, eid)
            for k, v in unnorm.items():
                unnorm_preds[f"{embodiment_name}_{k}"] = v
        return unnorm_preds

    def forward_eval_logging(self, batch):
        preds = self.forward_eval(batch)
        metrics: dict[str, float] = {}
        images: dict = {}
        mse = MeanSquaredError()
        stage = self.current_stage()
        for eid, _batch in batch.items():
            unbatch = self.data_schematic.unnormalize_data(_batch, eid)
            embodiment_name = get_embodiment(eid).lower()
            if stage == 1:
                target_key = self.shared_ac_key_stage1
            else:
                target_key = self.shared_ac_key_stage2
            pk = f"{embodiment_name}_{target_key}"
            if pk in preds and target_key in unbatch:
                gt = unbatch[target_key]
                if gt.ndim == 2:
                    gt = gt.unsqueeze(1)
                metrics[f"Valid/{pk}_paired_mse_avg"] = mse(
                    preds[pk].cpu(), gt.cpu()
                ).item()
            images[eid] = None
        return metrics, images

    def visualize_preds(self, predictions, batch):
        return None
