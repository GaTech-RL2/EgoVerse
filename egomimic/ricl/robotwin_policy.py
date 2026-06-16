"""EgoVerse RICL policy for RoboTwin's closed-loop eval (the source of truth).

RoboTwin's eval driver loads a policy by ``importlib.import_module(policy_name)`` and
calls module-level ``get_model`` / ``eval`` / ``reset_model`` (and the policy object's
``set_language`` / ``update_observation_window`` / ``get_action`` /
``reset_obsrvationwindows``). That contract lives in the RoboTwin tree at
``policy/pi_ricl_egoverse/`` — but only as a **thin shim** that re-exports this module
(``from egomimic.ricl.robotwin_policy import *``), so the actual logic stays here in
egomimic next to the adapter glue (``robotwin_adapter``) and stays unit-tested.

This wraps the EgoVerse PyTorch ``PIRicl`` policy with online RICL retrieval (vs the
stock pi05 openpi-JAX path). Per inference: embed the live head frame -> kNN against a
prebuilt demo bank -> splice the retrieved (image, state, action) blocks into the pi0.5
prefix -> sample an action chunk -> un-normalize the model's 32-D output back to
RoboTwin's native qpos. torch + the EgoVerse algo + DINOv2 are imported lazily so this
module imports without a GPU/checkpoint (the model-free glue is exercised by
``egomimic/ricl/tests/robotwin_adapter_test.py``). The PIRicl construction mirrors
``egomimic/ricl/scripts/train_robotwin_ricl.build_model``; verify the checkpoint-load +
``forward_eval`` input contract on first deploy.

``usr_args`` keys (set in the shim's ``deploy_policy.yml`` + eval.sh overrides):
    egoverse_checkpoint : path to the trained Lightning ``.ckpt``
    bank_root           : RoboTwin task dir whose demos form the retrieval bank
    bank_index_dir      : consolidated DINOv2 index for ``bank_root`` (build_embedding_index)
    quantiles_path      : norm-stats JSON saved at training (RoboTwinCorpus.save_quantiles)
    k, action_horizon, pi0_step : retrieval/inference sizes
"""

from __future__ import annotations

import os

import numpy as np

# Vendored (gitignored) PaliGemma tokenizer — same one training uses; avoids a
# gated HF download on the eval node. Override via usr_args["tokenizer"].
DEFAULT_TOKENIZER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pg_tokenizer"
)

__all__ = [
    "encode_obs",
    "get_model",
    "eval",
    "reset_model",
    "PIRiclPolicy",
]


# --------------------------------------------------------------------------- #
# RoboTwin deploy contract (module-level functions the eval driver calls)
# --------------------------------------------------------------------------- #
def encode_obs(observation):
    """RoboTwin obs -> ([head, right, left] rgb, qpos vector). Mirrors pi05."""
    from egomimic.ricl.robotwin_adapter import obs_to_state

    return obs_to_state(observation)


def get_model(usr_args):
    return PIRiclPolicy(usr_args)


def eval(TASK_ENV, model, observation):  # noqa: A001 (name required by RoboTwin driver)
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[: model.pi0_step]
    for action in actions:
        TASK_ENV.take_action(action)  # default qpos control mode (joint-space)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()


# --------------------------------------------------------------------------- #
# Policy object
# --------------------------------------------------------------------------- #
class PIRiclPolicy:
    """RoboTwin ``PI0``-shaped wrapper around EgoVerse ``PIRicl`` + online retrieval."""

    def __init__(self, usr_args: dict):
        import torch  # noqa: F401  (asserts the torch env exists)

        from egomimic.ricl import robotwin_adapter as A
        from egomimic.ricl import robotwin_data as R
        from egomimic.ricl.retrieval import DinoV2Embedder
        from egomimic.ricl.scripts.build_embedding_index import build_retrieval_index

        self._A = A
        self.pi0_step = int(usr_args.get("pi0_step", 50))
        self.action_horizon = int(usr_args.get("action_horizon", 15))
        self.k = int(usr_args.get("k", 4))
        self.tokenizer_dir = usr_args.get("tokenizer") or DEFAULT_TOKENIZER_DIR

        # Norm quantiles from training (so eval normalizes identically).
        self.quantiles = R.load_quantiles(usr_args["quantiles_path"])

        # Demo bank: a RoboTwin corpus + its consolidated DINOv2 index, queried online.
        bank_corpus = R.RoboTwinCorpus(usr_args["bank_root"], quantiles=self.quantiles)
        self.state_dim = bank_corpus.state_dim
        self.gripper_slots = bank_corpus.gripper_slots
        bank_provider = R.make_robotwin_bank_provider(
            bank_corpus, action_horizon=self.action_horizon
        )
        index = build_retrieval_index(usr_args["bank_index_dir"])
        self.retriever = A.OnlineRetriever(
            index,
            bank_provider,
            DinoV2Embedder(),
            k=self.k,
            action_horizon=self.action_horizon,
            state_dim=self.state_dim,
        )

        self.algo = self._load_algo(usr_args["egoverse_checkpoint"])
        self.observation_window = None
        self.instruction = None

    def _load_algo(self, ckpt_path: str):
        """Reconstruct PIRicl and load the trained Lightning checkpoint weights.

        Mirrors ``train_robotwin_ricl.build_model`` for the architecture, then loads
        the checkpoint's ``state_dict`` (the ckpt is a ModelWrapper, so weights are
        under the ``model.`` prefix). Verify on first deploy."""
        from types import SimpleNamespace

        import torch

        from egomimic.algo.pi_ricl import PIRicl
        from egomimic.ricl import robotwin_data as R

        config = SimpleNamespace(
            pytorch_training_precision="bfloat16",
            pytorch_weight_path=None,  # weights come from the trained ckpt below
            model=SimpleNamespace(
                pi05=True,
                action_dim=32,
                action_horizon=self.action_horizon,
                max_token_len=4608,
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
            ),
        )
        converters = SimpleNamespace(
            rules={R.EMB_NAME: R.Passthrough32()},
            fallback=R.Passthrough32(),
            ac_key=R.AC_KEY,
        )
        algo = PIRicl(
            norm_stats=R.RoboTwinNormStats(),
            camera_transforms=None,
            domains=[R.EMB_NAME],
            train_image_augs=None,
            eval_image_augs=None,
            config=config,
            ac_keys={R.EMB_NAME: R.AC_KEY},
            action_converters=converters,
            tokenizer_model_name=self.tokenizer_dir,
            tokenizer_max_length=4608,
            sampling_mode="random",
            annotation_key="annotations",
            default_prompt="",
            proprio_in_prompt=True,
            embodiment_label=True,
            state_num_bins=256,
            proprio_keys_for_prompt=[R.PROMPT_STATE_KEY],
            num_retrieved_observations=self.k,
            retrieved_action_steps=self.action_horizon,
            ricl_base_key="base_0_rgb",
            image_resolution=(224, 224),
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # ModelWrapper stores the algo under ``model.``; strip that prefix.
        weights = {
            k[len("model.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        } or state_dict
        algo.nets["policy"].load_state_dict(weights, strict=False)
        algo.eval()
        return algo

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction

    def update_observation_window(self, img_arr, state) -> None:
        # img_arr = [head, right, left] (from encode_obs); state = raw qpos vector.
        self.observation_window = {
            "base_0_rgb": np.asarray(img_arr[0]),
            "right_wrist_0_rgb": np.asarray(img_arr[1]),
            "left_wrist_0_rgb": np.asarray(img_arr[2]),
            "state": np.asarray(state, dtype=np.float32),
        }

    def get_action(self):
        """Sample an action chunk and map it to RoboTwin qpos (pi0_step, state_dim)."""
        import torch

        from egomimic.ricl import robotwin_data as R

        assert self.observation_window is not None, "update observation_window first!"
        w = self.observation_window
        A = self._A

        ricl = self.retriever.retrieve(w["base_0_rgb"])  # ricl_retrieved_* (batch 1)
        state32 = A.state_to_model_input(w["state"], self.quantiles["state"])
        raw = {
            R.EMB_NAME: {
                "base_0_rgb": torch.as_tensor(w["base_0_rgb"])[None],
                "left_wrist_0_rgb": torch.as_tensor(w["left_wrist_0_rgb"])[None],
                "right_wrist_0_rgb": torch.as_tensor(w["right_wrist_0_rgb"])[None],
                R.STATE_KEY: torch.as_tensor(state32, dtype=torch.float32)[None],
                "annotations": [[self.instruction or ""]],
                "episode_hash": ["__live__"],
                "frame_idx": [0],
                **ricl,
            }
        }
        with torch.no_grad():
            proc = self.algo.process_batch_for_training(raw)
            preds = self.algo.forward_eval(proc)
        pred = preds[f"{R.EMB_NAME}_{R.AC_KEY}"][0].detach().cpu().numpy()  # (H, 32)
        return A.unnormalize_action(pred, self.quantiles["actions"], self.state_dim)

    def reset_obsrvationwindows(self) -> None:
        self.instruction = None
        self.observation_window = None
