# ruff: noqa: E402
"""Arc-tokenized variant of ``egomimic/robot/rollout.py``.

What's different from ``rollout.py``:

    * ``rollout.py`` assumes the loaded policy emits a time-parameterized
      cartesian action chunk ``(H, 14)`` that the controller can consume
      row-by-row at the control period. This script adapts the "predict ->
      send to robot" step for policies trained with the arc-length tokenizer
      (see ``egomimic/rldb/zarr/arc_length_tokenizer.py`` and the model
      configs ``hpt_cotrain_mecka_flow_shared_head_arc_D40_M100*.yaml``).
    * Arc-tok models output ``(B, M+1, 14)`` per sample: M waypoints uniform
      in each arm's arc length + 1 velocity token, each 14-dim
      ``[L xyz, L ypr, L grip, R xyz, R ypr, R grip]``. Rotation is
      unconditionally supervised (no drop-rotation option).
    * Before the raw prediction goes to the controller, we DETOKENIZE it
      into a canonical ``(H, 14)`` cartesian chunk via
      ``TokenizeBimanualArcLengthCartesian.detokenize`` -- the exact same
      reconstruction algebra ``egomimic/eval/eval_arctok.py`` uses for val
      videos (xyz linear, ypr SLERP, gripper linear along per-arm xyz
      cumdist). Duration is derived from the velocity token's xyz slots;
      degenerate arms hold pose (matching the tokenizer/eval behaviour).
    * Everything else (obs-batch construction, embodiment routing, control
      loop, safety limits, interventions, resampling, cam-to-base
      transforms) is preserved unchanged from ``rollout.py`` so the two
      paths stay side-by-side comparable.

Borrowed from:
    * ``egomimic/robot/rollout.py`` -- 100% of the scaffold: PolicyRollout
      structure, obs preprocessing, cartesian rot/frame handling, safety
      resampling, terminal intervention loop, argparse.
    * ``egomimic/eval/eval_arctok.py`` (``ArcTokEvalVideo._detokenize_batch``)
      -- the (B, M+1, 14) -> (B, H, 14) detokenization pass wrapped around
      ``TokenizeBimanualArcLengthCartesian.detokenize``.
    * ``egomimic/rldb/zarr/arc_length_tokenizer.py``
      (``TokenizeBimanualArcLengthCartesian``) -- the underlying detokenize
      implementation.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Can't initialize NVML")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARC_TOK_BIMANUAL_DIM,
    TokenizeBimanualArcLengthCartesian,
)
from egomimic.rldb.zarr.e1_arc_tokenizer import (
    E1_ARCVEL_DIM,
    TokenizeBimanualArcLengthE1,
)

# Token layouts this script can send to the robot.
#   lab         (M+1, 14) chord-norm arc token + trailing velocity row
#   e1_dur      (M, 16)   waypoints + absolute interval seconds per arm
#   e1_logdur   (M, 16)   waypoints + log mean slowness / log relative durations
#   e1_profile  (M, 16)   waypoints + per-waypoint speed (Arc+Vel)
# The (M,16) rows carry timing in COLUMNS rather than a trailing row, so they
# are M tokens wide, not M+1 -- every shape check below is layout-aware.
E1_VELOCITY_MODE = {"e1_dur": "dur", "e1_logdur": "logdur", "e1_profile": "profile"}
ARC_TOKEN_LAYOUTS = ("lab",) + tuple(E1_VELOCITY_MODE)
from egomimic.robot.robot_utils import RateLoop
from egomimic.robot.rollout import (
    PolicyRollout,
    ReplayRollout,
    _build_robot_interface,
    _KeyPoll,
    build_arg_parser,
    debug_policy,
    reset_rollout,
    rot_ee_frame_to_ee_pose_batch,
)
from egomimic.utils.pose_utils import cam_frame_to_base_frame

sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

import termios
import tty

# Defaults match ``arc_sweep_D40_M100.yaml`` / the D40_M100 model family
# -- the current headline arc-tok configuration. Override via CLI if a
# different D/M ckpt is loaded (the checkpoint's config_tree only carries
# the model subtree, not the data pipeline's tokenizer knobs).
DEFAULT_ARC_MIN_DISTANCE_UNIT = 0.40  # D in meters
DEFAULT_ARC_RESAMPLED_VECTOR_LENGTH = 100  # M waypoints
DEFAULT_ARC_ROLLOUT_HORIZON = 100  # H control-period steps to reconstruct
DEFAULT_ARC_DT = 1.0 / 30.0  # control period


class ArcTokPolicyRollout(PolicyRollout):
    """PolicyRollout specialized for arc-tokenized policies.

    Loads the checkpoint the same way ``PolicyRollout`` does (reuses
    ``_load_policy`` + ``_patch_checkpoint_paths`` from the base class),
    processes the observation into the same batch dict (the arc-tok data
    pipeline uses the same obs keys as the non-arc pipeline -- only the
    ACTION side is different), and then wraps the base-class prediction
    step with a detokenize pass:

        raw model out  (B, M+1, 14)  <- forward_eval
                v                     TokenizeBimanualArcLengthCartesian.detokenize
        detok chunk    (B, H, 14)     canonical [L xyz ypr grip | R xyz ypr grip]
                v                     hand to rollout.py's cam->base + safety
                                      resample path unchanged
        controller     (H, 14)        row-per-control-step, arm-split before
                                      set_pose / set_joints
    """

    def __init__(
        self,
        arm,
        policy_path,
        query_frequency,
        cartesian,
        resampled_action_len=None,
        debug=False,
        annotation_path=None,
        min_distance_unit=DEFAULT_ARC_MIN_DISTANCE_UNIT,
        resampled_vector_length=DEFAULT_ARC_RESAMPLED_VECTOR_LENGTH,
        rollout_horizon=DEFAULT_ARC_ROLLOUT_HORIZON,
        arc_dt=DEFAULT_ARC_DT,
        token_layout="lab",
    ):
        super().__init__(
            arm=arm,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            resampled_action_len=resampled_action_len,
            debug=debug,
            annotation_path=annotation_path,
        )
        # Try to pull M+1 (= act_seq / action_horizon) from the loaded
        # checkpoint's saved model config. This lets us warn early if the
        # CLI-provided M doesn't match what the model actually emits.
        # D itself is NOT recoverable from the checkpoint (it lives in the
        # data pipeline's tokenizer transform, which is not serialised into
        # config_tree) -- must be passed on the CLI to match training.
        if token_layout not in ARC_TOKEN_LAYOUTS:
            raise ValueError(
                f"[rollout-arc] --arc-token-layout must be one of {ARC_TOKEN_LAYOUTS}, "
                f"got {token_layout!r}"
            )
        self._token_layout = token_layout
        self._token_rows = int(resampled_vector_length) + (1 if token_layout == "lab" else 0)
        self._token_dim = ARC_TOK_BIMANUAL_DIM if token_layout == "lab" else E1_ARCVEL_DIM
        M_plus_1 = self._read_action_horizon_from_ckpt(policy_path)
        if M_plus_1 is not None and M_plus_1 != self._token_rows:
            print(
                f"[rollout-arc] WARNING: checkpoint act_seq={M_plus_1} does not "
                f"match --arc-resampled-vector-length={resampled_vector_length} "
                f"for layout {token_layout!r} (expected {self._token_rows}). The "
                f"detokenize call will raise on a shape mismatch."
            )

        # Same class ``ArcTokEvalVideo`` uses for val videos (see
        # ``eval_arctok.py:81``) -- keep deploy identical to eval.
        if token_layout == "lab":
            self._arc_detokenizer = TokenizeBimanualArcLengthCartesian(
                action_key="actions_cartesian",
                output_action_key="actions_cartesian",
                min_distance_unit=float(min_distance_unit),
                resampled_vector_length=int(resampled_vector_length),
                dt=float(arc_dt),
            )
        else:
            # velocity_norm="path" matches how the E1 rows were tokenized at
            # training time; the lab token is chord-normed and keeps its own class.
            self._arc_detokenizer = TokenizeBimanualArcLengthE1(
                action_key="actions_cartesian",
                output_action_key="actions_cartesian",
                min_distance_unit=float(min_distance_unit),
                resampled_vector_length=int(resampled_vector_length),
                dt=float(arc_dt),
                velocity_norm="path",
                velocity_mode=E1_VELOCITY_MODE[token_layout],
            )
        self._M = int(resampled_vector_length)
        self._arc_rollout_horizon = int(rollout_horizon)
        self._arc_dt = float(arc_dt)
        print(
            f"[rollout-arc] Arc detokenizer: layout={token_layout} "
            f"D={min_distance_unit}m M={resampled_vector_length} "
            f"H={rollout_horizon} dt={arc_dt:.5f}s "
            f"(expects ({self._token_rows}, {self._token_dim}) per sample)"
        )

    @staticmethod
    def _read_action_horizon_from_ckpt(ckpt_path):
        """Return the model's ``action_horizon`` (M+1) from the checkpoint
        or None if it isn't recoverable. Mirrors the config_tree access
        path used by ``PolicyRollout._patch_checkpoint_paths``."""
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[rollout-arc] Could not read ckpt for act_seq check: {e}")
            return None
        ht = ckpt.get("hyper_parameters", {}).get("config_tree")
        if ht is None:
            return None
        cfg = (
            OmegaConf.to_container(ht, resolve=True)
            if isinstance(ht, DictConfig)
            else ht
        )
        try:
            head_specs = cfg["model"]["robomimic_model"]["head_specs"]
            shared = head_specs.get("shared", {})
            ah = shared.get("action_horizon")
            if ah is not None:
                return int(ah)
        except (KeyError, TypeError):
            pass
        return None

    def _detokenize_arc_output(self, arc_out):
        """(B, M+1, 14) -> (B, H, 14) canonical bimanual cartesian.

        The arc-tok head now unconditionally supervises orientation, so
        the detokenizer returns the full 14-dim ``[L xyz ypr grip | R xyz
        ypr grip]`` layout directly -- no zero-padding of rotation columns
        is needed. This mirrors ``ArcTokEvalVideo._detokenize_batch``.
        """
        if not isinstance(arc_out, torch.Tensor):
            arc_out = torch.as_tensor(arc_out)
        arc_np = arc_out.detach().cpu().numpy().astype(np.float64)
        if arc_np.ndim == 2:
            arc_np = arc_np[None, ...]  # (M+1, 14) -> (1, M+1, 14)
        if arc_np.ndim != 3 or arc_np.shape[-1] != self._token_dim:
            raise ValueError(
                f"[rollout-arc] layout={self._token_layout} expects arc output "
                f"(B, {self._token_rows}, {self._token_dim}), got {arc_np.shape}"
            )
        if arc_np.shape[1] != self._token_rows:
            raise ValueError(
                f"[rollout-arc] layout={self._token_layout} configured for M={self._M} "
                f"({self._token_rows} token rows), got {arc_np.shape[1]} rows in the "
                f"model output"
            )
        B = arc_np.shape[0]
        H = self._arc_rollout_horizon
        out = np.zeros((B, H, 14), dtype=np.float64)
        for b in range(B):
            # (H, 14) directly -- rotation is unconditionally supervised
            # and returned from the detokenizer, so no post-splice is needed.
            out[b] = self._arc_detokenizer.detokenize(arc_np[b], action_horizon=H)
        return out

    def rollout_step(self, i, obs):
        """Predict -> DETOKENIZE -> cam-to-base + resample -> emit one row.

        This is a copy of ``PolicyRollout.rollout_step`` with a single
        surgical change: after ``forward_eval`` returns the raw arc-tok
        tensor, we detokenize it BEFORE the cartesian rot/frame transforms
        + safety resample fire. All safety-relevant code paths (rot
        transforms, cam_frame_to_base_frame, ``_downsample_chunk``) then
        operate on the DETOKENIZED (H, 14) chunk exactly as they would for
        a non-arc model -- rollout.py's controller invariants are preserved.
        """
        if i % self.query_frequency == 0:
            start_infer_t = time.time()
            transform_list_batch = self.process_obs_for_transform_list(obs)
            for transform in self.transform_list:
                transform_list_batch = transform.transform(transform_list_batch)
            transform_list_batch = self.collate_fn([transform_list_batch])
            if self.arm == "both":
                embodiment_name = "eva_bimanual"
            elif self.arm == "right":
                embodiment_name = "eva_right_arm"
            elif self.arm == "left":
                embodiment_name = "eva_left_arm"
            batch = {
                embodiment_name: transform_list_batch,
            }
            processed_batch = self.policy.model.process_batch_for_training(batch)
            preds = self.policy.model.forward_eval(processed_batch)[
                f"{embodiment_name}_actions_cartesian"
            ]

            # -- ARC-TOK DIFF: raw preds here are (B, M+1, 14) instead of
            # the non-arc (B, H, 14). Detokenize BEFORE any downstream
            # safety / cam-frame / resample step so those operate on the
            # same shape rollout.py assumes.
            raw_arc = preds.detach().cpu().numpy()
            print(
                f"[rollout-arc] raw arc preds shape={raw_arc.shape}  "
                f"(expected (B, {self._M + 1}, {ARC_TOK_BIMANUAL_DIM}))"
            )
            detok = self._detokenize_arc_output(preds)  # (B, H, 14)
            # Match the base class's contract: strip the batch dim (single
            # sample per inference call at deploy time) so ``self.actions``
            # ends up (H, 14) exactly like PolicyRollout's non-arc path.
            self.actions = detok.astype(np.float32, copy=False).squeeze(axis=0)
            self.debug_actions = self.actions.copy()
            # --

            # Everything below mirrors PolicyRollout.rollout_step's post-
            # prediction path unchanged so cam->base + safety resample fire
            # on the DETOKENIZED chunk.
            if self.cartesian:
                if self.arm == "both":
                    left_actions = self.actions[:, :7]
                    right_actions = self.actions[:, 7:]

                    transformed_left = cam_frame_to_base_frame(
                        left_actions[:, :6].copy(), self.extrinsics["left"]
                    )
                    transformed_right = cam_frame_to_base_frame(
                        right_actions[:, :6].copy(), self.extrinsics["right"]
                    )
                    transformed_left = rot_ee_frame_to_ee_pose_batch(transformed_left)
                    transformed_right = rot_ee_frame_to_ee_pose_batch(transformed_right)
                    gripper_left = left_actions[:, 6:7]
                    gripper_right = right_actions[:, 6:7]
                    if left_actions.shape[1] == 7:
                        left_actions = np.hstack([transformed_left, gripper_left])
                    else:
                        left_actions = transformed_left
                    if right_actions.shape[1] == 7:
                        right_actions = np.hstack([transformed_right, gripper_right])
                    else:
                        right_actions = transformed_right
                    self.actions = np.hstack([left_actions, right_actions])
                else:
                    eepose = rot_ee_frame_to_ee_pose_batch(self.actions[:, :6].copy())
                    self.actions[:, :6] = eepose
                    transformed_6dof = cam_frame_to_base_frame(
                        self.actions[:, :6].copy(), self.extrinsics[self.arm]
                    )
                    # Preserve gripper if present (7th value)
                    gripper = self.actions[:, 6:7]
                    if self.actions.shape[1] == 7:
                        self.actions = np.hstack([transformed_6dof, gripper])
                    else:
                        self.actions = transformed_6dof

            if self.resampled_action_len is not None:
                self.actions = self._downsample_chunk(
                    self.actions, self.resampled_action_len
                )

            print(f"Inference time: {(time.time() - start_infer_t)}s")

        act_i = i % self.query_frequency
        return self.actions[act_i]


def main(
    arms,
    frequency,
    cartesian,
    query_frequency=None,
    policy_path=None,
    dataset_path=None,
    debug=False,
    resampled_action_len=None,
    offline_debug=False,
    offline_episode_path=None,
    annotation_path=None,
    arc_min_distance_unit=DEFAULT_ARC_MIN_DISTANCE_UNIT,
    arc_resampled_vector_length=DEFAULT_ARC_RESAMPLED_VECTOR_LENGTH,
    arc_rollout_horizon=DEFAULT_ARC_ROLLOUT_HORIZON,
    arc_dt=DEFAULT_ARC_DT,
    arc_token_layout="lab",
):
    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    if offline_episode_path is not None and not offline_debug:
        raise ValueError("--offline-episode-path requires --offline-debug.")
    if policy_path is not None and offline_debug and offline_episode_path is None:
        raise ValueError(
            "--policy-path requires --offline-episode-path in --offline-debug mode."
        )

    ri = _build_robot_interface(
        arms_list=arms_list,
        offline_debug=offline_debug,
        offline_episode_path=offline_episode_path,
    )

    if policy_path is not None:
        rollout_type = "policy"
        policy = ArcTokPolicyRollout(
            arm=arms,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            resampled_action_len=resampled_action_len,
            debug=debug,
            annotation_path=annotation_path,
            min_distance_unit=arc_min_distance_unit,
            resampled_vector_length=arc_resampled_vector_length,
            rollout_horizon=arc_rollout_horizon,
            arc_dt=arc_dt,
            token_layout=arc_token_layout,
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise ValueError(
            "Must provide either --policy-path or --dataset-path (and optionally --repo-id)."
        )

    print(f"Cartesian value {cartesian}")

    def _enter_intervention(kp, policy, rollout_type):
        """Pause rollout and wait for user command.

        Restores the terminal to cooked mode so the user can type full
        commands, then re-enters cbreak mode before returning.

        Returns one of:
            "continue"  - resume rollout
            "restart"   - restart rollout
            "quit"      - exit program
        """
        termios.tcsetattr(kp.fd, termios.TCSADRAIN, kp.old)
        print("\n--- INTERVENTION (rollout paused) ---")
        print("  c            : continue rollout")
        print("  a <path>     : load new annotation file")
        print("  r            : restart rollout")
        print("  q            : quit")

        while True:
            try:
                cmd = input("> ").strip()
            except EOFError:
                tty.setcbreak(kp.fd)
                return "quit"

            if cmd == "c":
                print("Resuming rollout.")
                tty.setcbreak(kp.fd)
                return "continue"
            elif cmd == "q":
                tty.setcbreak(kp.fd)
                return "quit"
            elif cmd == "r":
                tty.setcbreak(kp.fd)
                return "restart"
            elif cmd.startswith("a "):
                ann_path = cmd[2:].strip()
                if not ann_path:
                    print("Usage: a <annotation_path>")
                    continue
                if rollout_type != "policy" or not isinstance(policy, PolicyRollout):
                    print("Annotation loading is only supported for policy rollouts.")
                    continue
                policy.load_annotation(ann_path)
            else:
                print(f"Unknown command: '{cmd}'. Use c / a <path> / r / q.")

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)
            result = _enter_intervention(kp, policy, rollout_type)
            if result == "quit":
                print("Quit requested.")
                return
            if result == "restart":
                reset_rollout(ri, policy)

            while True:  # restartable
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch is not None:
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                print("Quit requested.")
                                return
                            elif result == "restart":
                                print("Restart requested.")
                                reset_rollout(ri, policy)
                                result = _enter_intervention(kp, policy, rollout_type)
                                if result == "quit":
                                    return
                                if result == "restart":
                                    reset_rollout(ri, policy)
                                break
                            if hasattr(policy, "actions"):
                                policy.actions = None
                            break

                        actions = None
                        if rollout_type == "policy":
                            obs = ri.get_obs()
                            actions = policy.rollout_step(step_i, obs)
                        elif rollout_type == "replay":
                            actions = policy.rollout_step(step_i)
                        else:
                            raise ValueError(f"Invalid rollout type: {rollout_type}")

                        if actions is None:
                            print("Finish rollout.")
                            reset_rollout(ri, policy)
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                return
                            if result == "restart":
                                reset_rollout(ri, policy)
                            break

                        if (
                            debug
                            and rollout_type == "policy"
                            and step_i % query_frequency == 0
                        ):
                            debug_actions = policy.debug_actions
                            front_img = obs["front_img_1"]
                            debug_policy(
                                debug_actions,
                                front_img,
                                step_i,
                            )

                        for arm in arms_list:
                            arm_offset = 7 if (arm == "right" and arms == "both") else 0
                            arm_action = actions[arm_offset : arm_offset + 7]
                            if cartesian:
                                ri.set_pose(arm_action, arm)
                            else:
                                ri.set_joints(arm_action, arm)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting rollout.")
        return


def build_arc_arg_parser():
    """Extends ``rollout.build_arg_parser`` with arc-tok knobs.

    The base ``rollout.py`` parser handles all the shared flags (arms,
    frequency, policy-path, dataset-path, cartesian, resampled-action-len,
    debug, annotation-path, offline-debug, offline-episode-path). We layer
    in D / M / H / dt for the arc detokenizer. Values MUST match the ones
    the data pipeline used to produce the training targets (see the
    ``arc_sweep_D*_M*.yaml`` data configs); the checkpoint's config_tree
    only carries the model subtree, not the tokenizer knobs.
    """
    parser = build_arg_parser(
        description="Rollout arc-tokenized robot policy (see rollout-arc.py)."
    )
    parser.add_argument(
        "--arc-min-distance-unit",
        type=float,
        default=DEFAULT_ARC_MIN_DISTANCE_UNIT,
        help=(
            "D (meters) -- per-arm arc-length span of one arc token. Must "
            "match the value the training data pipeline used."
        ),
    )
    parser.add_argument(
        "--arc-resampled-vector-length",
        type=int,
        default=DEFAULT_ARC_RESAMPLED_VECTOR_LENGTH,
        help=(
            "M -- number of arc waypoints (model must emit (M+1, 14) per "
            "sample; must match training-time value)."
        ),
    )
    parser.add_argument(
        "--arc-rollout-horizon",
        type=int,
        default=DEFAULT_ARC_ROLLOUT_HORIZON,
        help=(
            "H -- number of control-period steps to reconstruct per chunk. "
            "Matches ``rollout_horizon`` in eval_arctok.yaml."
        ),
    )
    parser.add_argument(
        "--arc-dt",
        type=float,
        default=DEFAULT_ARC_DT,
        help="Control period (seconds) used by the arc detokenizer.",
    )
    parser.add_argument(
        "--arc-token-layout",
        choices=ARC_TOKEN_LAYOUTS,
        default="lab",
        help=(
            "Token layout the checkpoint emits. 'lab' is the (M+1,14) chord-norm "
            "token (default, unchanged behaviour); the e1_* layouts are (M,16) "
            "with timing in per-arm columns."
        ),
    )
    return parser


def run_from_args(args):
    print(f"Resampling actions to {args.resampled_action_len}")
    return main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
        debug=args.debug,
        resampled_action_len=args.resampled_action_len,
        offline_debug=args.offline_debug,
        offline_episode_path=args.offline_episode_path,
        annotation_path=args.annotation_path,
        arc_min_distance_unit=args.arc_min_distance_unit,
        arc_resampled_vector_length=args.arc_resampled_vector_length,
        arc_rollout_horizon=args.arc_rollout_horizon,
        arc_dt=args.arc_dt,
        arc_token_layout=args.arc_token_layout,
    )


if __name__ == "__main__":
    parser = build_arc_arg_parser()
    args = parser.parse_args()
    run_from_args(args)
