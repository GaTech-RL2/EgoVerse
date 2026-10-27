"""Mixture-of-frames flow head (user 2026-10-17: "both the actions and observations ... in many different frames").

One flow-matching expert per frame. Every expert predicts the same 67-D chunk (the 49-D whole-body joints + the left and right
eef target poses, pos + rot6d, of the eefb action column) with the two eef targets expressed in ITS frame, fixed at the
observation row: the robot base, the robot camera (glass), the current left eef, the current right eef. The frames come from
FK of the row's 22-D proprio (the clean value, stashed by HPTModel before the stems add noise / dropout). A gate on the trunk
features weights the experts separately for the joints, the left eef and the right eef.

Training: each expert's own flow loss in its own frame + a gate loss on the experts' (detached) one-step estimates mapped to
the base frame and mixed, weighted (1 - t)^2 toward the clean end. Inference: each expert integrates its own ODE (with RTC
inpainting per expert when the server arms _rtc_ctx), the samples are mapped to the base frame and mixed by the gate. The
output is the eefb chunk (normalized like the column), so deployment is the eef IK server unchanged (frame base).

Expert spaces: the joints stay normalized by the schematic; eef positions are metres / POS_SCALE and the rot6d columns are
left as they are, so every frame shares one scale and no per-frame statistics are needed.
"""
import copy
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.denoising_policy import DenoisingPolicy

SEW = "/coc/flash7/czhang883/Documents/SEW-Geometric-Teleop-hw"
POS_SCALE = 0.5            # m
N_JOINT = 49               # [0:49] joints | [49:58] left eef p c1 c2 | [58:67] right eef p c1 c2
GROUP = [0] * 49 + [1] * 9 + [2] * 9   # gate group of each action dim: joints / left eef / right eef
IS_POS = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])   # of the six 3-vectors in the 18 eef dims, which are positions


def _quat2mat(q):  # MuJoCo (w, x, y, z)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class RBY1TorchFK(nn.Module):
    """Batched torch FK of model_v1.3_xhand_act.xml (the dataset builder's and the eef IK server's model), relative to body
    'base': q22 [torso 6 | r_arm 7 | l_arm 7 | head 2] -> T_base_glass, T_base_eefL, T_base_eefR. The chains are read
    from the MuJoCo model with MuJoCo's own rule (body offset, then each hinge rotated by q - qpos0 about its anchor);
    glass = link_head_2 @ M_HEAD_DEV as in RBY1EefFK."""

    TARGETS = {"glass": "link_head_2", "eefL": "left_eef", "eefR": "right_eef"}

    def __init__(self):
        super().__init__()
        import mujoco

        sys.path.insert(0, SEW)
        from projects.rby1_teleop.utils import rby1_eef_frames as ref

        m = mujoco.MjModel.from_xml_path(str(ref.DEFAULT_MODEL_XML))
        qidx = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n): i for i, n in enumerate(ref._Q22_JOINTS)}
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.qi = {}
        for frame, body in self.TARGETS.items():
            b, path = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body), []
            while b != base:
                assert b > 0, f"{body} is not below body 'base'"
                path.append(b)
                b = int(m.body_parentid[b])
            Tfix, axis, anchor, q0, qi = [], [], [], [], []
            for b in reversed(path):
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = _quat2mat(m.body_quat[b]), m.body_pos[b]
                Tfix.append(T)
                js = range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
                assert len(js) <= 1, f"body {b}: {len(js)} joints"
                if len(js):
                    j = js[0]
                    assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and j in qidx, f"joint {j} on the {frame} chain"
                    axis.append(m.jnt_axis[j]), anchor.append(m.jnt_pos[j]), q0.append(m.qpos0[m.jnt_qposadr[j]])
                    qi.append(qidx[j])
                else:
                    axis.append(np.zeros(3)), anchor.append(np.zeros(3)), q0.append(0.0)
                    qi.append(-1)
            self.qi[frame] = qi
            for k, v in (("Tfix", Tfix), ("axis", axis), ("anchor", anchor), ("q0", q0)):
                self.register_buffer(f"{frame}_{k}", torch.tensor(np.array(v), dtype=torch.float64), persistent=False)
        self.register_buffer("M_HEAD_DEV", torch.tensor(ref.M_HEAD_DEV, dtype=torch.float64), persistent=False)

    @staticmethod
    def _hinge(axis, anchor, ang):
        """(B,) angles about one axis through anchor -> (B, 4, 4)"""
        k = axis / axis.norm()
        K = torch.zeros(3, 3, dtype=ang.dtype, device=ang.device)
        K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
        s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
        R = torch.eye(3, dtype=ang.dtype, device=ang.device) + s * K + (1 - c) * (K @ K)
        T = torch.zeros(len(ang), 4, 4, dtype=ang.dtype, device=ang.device)
        T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = R, anchor - R @ anchor, 1.0
        return T

    def forward(self, q):
        """q (B, 22) radians -> {frame: (B, 4, 4) T_base_F}, float64"""
        q = q.double()   # float64 throughout: exact, and outside bf16 autocast (HPT also .float()s the buffers)
        out = {}
        for frame in self.TARGETS:
            Tfix, axis, anchor, q0 = (getattr(self, f"{frame}_{k}").double() for k in ("Tfix", "axis", "anchor", "q0"))
            T = torch.eye(4, dtype=q.dtype, device=q.device).expand(len(q), 4, 4)
            for n, i in enumerate(self.qi[frame]):
                T = T @ Tfix[n]
                if i >= 0:
                    T = T @ self._hinge(axis[n], anchor[n], q[:, i] - q0[n])
            out[frame] = T
        out["glass"] = out["glass"] @ self.M_HEAD_DEV.double()
        return out


def _rigid_inv(T):
    R, t = T[..., :3, :3], T[..., :3, 3:]
    out = torch.zeros_like(T)
    out[..., :3, :3], out[..., :3, 3:], out[..., 3, 3] = R.transpose(-1, -2), -R.transpose(-1, -2) @ t, 1.0
    return out


def _apply18(T, x):
    """the 18 eef dims [pL c1L c2L pR c1R c2R] (B, H, 18) re-expressed by T (B, 4, 4): p -> R p + t, c -> R c
    (float64: autocast would run the einsum in bf16, ~3 mm at 1 m)"""
    v, T = x.double().reshape(*x.shape[:-1], 6, 3), T.double()
    v = torch.einsum("bij,bhkj->bhki", T[:, :3, :3], v) + IS_POS.to(v)[:, None] * T[:, None, None, :3, 3]
    return v.reshape(x.shape).to(x.dtype)


def _scale18(x, s):
    """positions * s, rot6d columns unchanged"""
    v = x.reshape(*x.shape[:-1], 6, 3)
    return (v * (1 + (s - 1) * IS_POS.to(x))[:, None]).reshape(x.shape)


class MoFFMPolicy(DenoisingPolicy):
    takes_data = True     # HPTModel stashes data["head_obs"] (the clean proprio) and passes the batch
    native_rtc = True     # the RTC server arms self._rtc_ctx; inference inpaints in every expert's frame
    obs_keys = ("state_robot0_joint_pos",)

    def __init__(self, model, action_horizon, infer_ac_dims, num_inference_steps=None,
                 frames=("base", "glass", "eefL", "eefR"), cond_dim=256, gate_hidden=256, gate_loss_weight=1.0, **kwargs):
        super().__init__(model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs)
        self.time_dist = kwargs.get("time_dist", "beta")
        self.frames = list(frames)
        assert set(self.frames) <= {"base", "glass", "eefL", "eefR"}, self.frames
        self.experts = nn.ModuleList([copy.deepcopy(model) for _ in self.frames])
        del self.model
        self.gate = nn.Sequential(nn.Linear(cond_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 3 * len(self.frames)))
        self.gate_loss_weight = float(gate_loss_weight)
        self.fk = RBY1TorchFK()
        self.register_buffer("group", torch.tensor(GROUP), persistent=False)
        self._rtc_ctx = None
        print(f"[MoFFMPolicy] {len(self.frames)} experts {self.frames}: "
              f"{sum(p.numel() for p in self.parameters()) / 1e6:.2f}M parameters")

    def set_norm(self, ac_stats, pr_stats):
        """quantile stats of the 67-D action column and the 22-D proprio (HPT init, from the schematic)"""
        for name, st in (("ac", ac_stats), ("pr", pr_stats)):
            self.register_buffer(f"{name}_lo", st["quantile_1"].float().clone(), persistent=False)
            self.register_buffer(f"{name}_rng", (st["quantile_99"] - st["quantile_1"]).float() + 1e-6, persistent=False)

    # ---- frames and spaces ----
    def frames_T(self, data):
        """(B, E, 4, 4) T_base_F of every expert frame at the observation row"""
        qn = data["head_obs"][self.obs_keys[0]].reshape(-1, 22).float()
        f = self.fk((qn + 1) * 0.5 * self.pr_rng + self.pr_lo)
        eye = torch.eye(4, dtype=torch.float64, device=qn.device).expand(len(qn), 4, 4)
        return torch.stack([eye if fr == "base" else f[fr] for fr in self.frames], 1).float()

    def to_phys(self, a):
        """schematic-normalized base-frame chunk -> joints normalized | eef in metres (base frame)"""
        e = (a[..., N_JOINT:] + 1) * 0.5 * self.ac_rng[N_JOINT:] + self.ac_lo[N_JOINT:]
        return torch.cat([a[..., :N_JOINT], e], -1)

    def from_phys(self, a):
        e = 2.0 * (a[..., N_JOINT:] - self.ac_lo[N_JOINT:]) / self.ac_rng[N_JOINT:] - 1.0
        return torch.cat([a[..., :N_JOINT], e], -1)

    def to_expert(self, a_phys, T_F_base):
        return torch.cat([a_phys[..., :N_JOINT], _scale18(_apply18(T_F_base, a_phys[..., N_JOINT:]), 1 / POS_SCALE)], -1)

    def from_expert(self, x, T_base_F):
        return torch.cat([x[..., :N_JOINT], _apply18(T_base_F, _scale18(x[..., N_JOINT:], POS_SCALE))], -1)

    def gate_w(self, cond):
        """(B, 67, E) expert weights of every action dim"""
        g = self.gate(cond.mean(1)).reshape(len(cond), 3, len(self.frames)).softmax(-1)
        return g[:, self.group]

    @staticmethod
    def mix(xs, w):
        return torch.einsum("bhde,bde->bhd", torch.stack(xs, -1).double(), w.double()).float()

    # ---- training ----
    def compute_loss(self, global_cond, data):
        actions, cond = self.preprocess_compute_loss(global_cond, data)
        Tbf = self.frames_T(data)
        Tfb = _rigid_inv(Tbf)
        a_phys = self.to_phys(actions.float())
        B = len(actions)
        if self.time_dist == "beta":
            time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(actions.device)
        else:
            time = torch.rand(B, device=actions.device)
        time = time * 0.999 + 0.001
        t = time[:, None, None]
        losses, est = [], []
        for e, net in enumerate(self.experts):
            tgt = self.to_expert(a_phys, Tfb[:, e])
            noise = torch.randn_like(tgt)
            x_t = t * noise + (1 - t) * tgt
            v = net(x_t, time, cond)
            losses.append(F.mse_loss(v, noise - tgt))
            est.append(self.from_expert((x_t - t * v).detach(), Tbf[:, e]))
        w = self.gate_w(cond)
        ref = self.to_expert(a_phys, torch.eye(4, device=a_phys.device).expand(B, 4, 4))   # base frame, expert scale
        mixed = self.to_expert(self.mix(est, w), torch.eye(4, device=a_phys.device).expand(B, 4, 4))
        tw = (1 - time) ** 2
        gate_loss = ((mixed - ref) ** 2).mean((1, 2)).mul(tw).sum() / tw.sum()
        loss = torch.stack(losses).mean() + self.gate_loss_weight * gate_loss
        bl = {f"expert_{fr}": l.detach() for fr, l in zip(self.frames, losses)}
        bl["gate"] = gate_loss.detach()
        for gi, gname in ((1, "l_eef"), (2, "r_eef"), (0, "joints")):
            wg = w[:, GROUP.index(gi)].mean(0).detach()
            bl.update({f"w_{gname}_{fr}": wg[e] for e, fr in enumerate(self.frames)})
        self.last_block_losses = bl
        return loss

    # ---- inference ----
    def forward(self, global_cond, data=None):
        cond, embodiment = global_cond
        _, cond = self.preprocess_sampling(cond, embodiment)
        return self.sample(cond, data)

    def sample(self, cond, data, generator=None):
        B, H, D = len(cond), self.action_horizon, N_JOINT + 18
        Tbf = self.frames_T(data)
        Tfb = _rigid_inv(Tbf)
        ctx = self._rtc_ctx
        if ctx is not None:   # RTC: the frozen / soft rows inpainted in every expert's own frame
            Y = ctx["target"].to(device=cond.device, dtype=cond.dtype).expand(B, H, D)
            wr = ctx["weights"].to(device=cond.device, dtype=cond.dtype)
            d = int(ctx["freeze"])
            Y_phys = self.to_phys(Y)
        dt = -1.0 / self.num_inference_steps
        outs = []
        for e, net in enumerate(self.experts):
            x = torch.randn((B, H, D), device=cond.device, dtype=cond.dtype, generator=generator)
            if ctx is not None:
                Ye, eps0 = self.to_expert(Y_phys, Tfb[:, e]), x
            time = torch.ones(B, device=cond.device)
            while time[0] >= -dt / 2:
                if ctx is not None:
                    x = wr * (time[0] * eps0 + (1.0 - time[0]) * Ye) + (1.0 - wr) * x
                x = x + dt * net(x, time, cond)
                time = time + dt
            if ctx is not None and d > 0:
                x = x.clone()
                x[:, :d] = Ye[:, :d]
            outs.append(self.from_expert(x, Tbf[:, e]))
        out = self.from_phys(self.mix(outs, self.gate_w(cond)))
        if ctx is not None and d > 0:
            out = out.clone()
            out[:, :d] = Y[:, :d]
        return out
