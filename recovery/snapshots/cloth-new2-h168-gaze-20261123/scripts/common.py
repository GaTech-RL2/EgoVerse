"""Cart gaze predictor (2026-10-27 copy of pipelines/fridge_tgz/gaze/common.py: only GZ and COLLECTIONS changed): shared constants, model, labels, per-episode cache, augmentation and metric.

Model = the deployed heat7 predictor: ResNet18 trunk (ImageNet init) -> 1x1 conv -> 7x7 heatmap -> soft-argmax (temp 2). Same module
tree as gaze_lab src/gazelab/models.HeatGaze and SEW utils/gaze_crop.HeatGaze._Net, so state_dict keys are body.*, head.*, gx, gy
(temp is a plain attribute, not a buffer). Input = rect RGB 224x224 as float in [0, 1], HWC -> CHW, NO ImageNet normalisation:
exactly what HeatGaze.predict_uv feeds the nets.

Label for row r of an episode (collection C, episode e = position of the demo in sorted(retargeted/C.hdf5 data keys)):
x = inv(tg[r]) @ [gp[r], 1] (gp = MPS gaze hit in the robot glass frame, tg = human rect camera -> robot glass, both written by
../real_dp3c_tri_gz.py); u = KH x0 / x2 + C0, v = KH x1 / x2 + C0 (the camera-centred projection of the look-at point).
Metric = angle between the pixel rays through two (u, v) with the same intrinsics (as pipelines/fridge_tgz/eval_gaze_fridge.py).
"""
import json, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W = Path("/coc/flash7/czhang883/warp-data")
GZ = Path(os.environ["GAZE_ROOT"])
DATA = GZ / "data"
COLLECTIONS = ("chuye_cart_1024", "chuye_cart_1027")
KH = 308.052437 * 224 / 512
C0 = 112.0
IMS = 224
SEW_GAZE_CROP = "/coc/flash7/czhang883/Documents/SEW-Geometric-Teleop-vv/projects/rby1_teleop/utils/gaze_crop.py"
OLD_ENS = Path("/coc/flash7/czhang883/handoff_gaze/egoverse_checkpoints_abl0921_gaze")


def coll_root(c):
    return W / f"collections/{c}/processed/r1"


def ep_key(c, e):
    return f"{c}:{int(e):06d}"


def parse_key(k):
    c, e = k.split(":")
    return c, int(e)


def ep_file(k):
    c, e = parse_key(k)
    return DATA / c / f"ep_{e:06d}.npz"


class Heat7(nn.Module):
    """= gazelab.models.HeatGaze(grid 7, temp 2) = gaze_crop.HeatGaze._Net (forward returns (B, 2) pixels)."""

    def __init__(self, pretrained=True, temp=2.0):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.body = nn.Sequential(*list(m.children())[:-2])          # (B, 512, 7, 7)
        self.head = nn.Conv2d(512, 1, 1)
        self.temp = temp
        gy, gx = torch.meshgrid(torch.arange(7.0), torch.arange(7.0), indexing="ij")
        self.register_buffer("gx", (gx + 0.5) / 7 * IMS)
        self.register_buffer("gy", (gy + 0.5) / 7 * IMS)

    def forward(self, x):
        h = self.head(self.body(x)).flatten(1)
        p = F.softmax(h * self.temp, dim=1)
        return torch.stack([(p * self.gx.flatten()).sum(1), (p * self.gy.flatten()).sum(1)], 1)


def project_labels(gp, tg):
    """gp (N,3) robot glass, tg (N,4,4) rect -> glass. Returns uv (N,2) and the rect-frame depth x2 (N,)."""
    xh = np.concatenate([np.asarray(gp, np.float64), np.ones((len(gp), 1))], 1)
    x = np.einsum("nij,nj->ni", np.linalg.inv(np.asarray(tg, np.float64)), xh)[:, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([KH * x[:, 0] / x[:, 2] + C0, KH * x[:, 1] / x[:, 2] + C0], 1)
    return uv, x[:, 2]


def in_image(uv):
    uv = np.asarray(uv)
    return np.all(np.isfinite(uv), 1) & (uv[:, 0] >= 0) & (uv[:, 0] < IMS) & (uv[:, 1] >= 0) & (uv[:, 1] < IMS)


def rays(uv):
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    d = np.stack([(uv[:, 0] - C0) / KH, (uv[:, 1] - C0) / KH, np.ones(len(uv))], 1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def ang_err(uv_a, uv_b):
    """Per-row angle (deg) between the pixel rays through uv_a and uv_b."""
    return np.degrees(np.arccos(np.clip((rays(uv_a) * rays(uv_b)).sum(1), -1.0, 1.0)))


def ang_err_t(pred, gt):
    """Torch version (= gaze_lab train_cache.ang_err_t)."""
    def ray(uv):
        x = (uv[:, 0] - C0) / KH
        y = (uv[:, 1] - C0) / KH
        d = torch.stack([x, y, torch.ones_like(x)], 1)
        return d / d.norm(dim=1, keepdim=True)
    return torch.rad2deg(torch.arccos((ray(pred) * ray(gt)).sum(1).clamp(-1, 1)))


def stats(e):
    e = np.asarray(e, np.float64)
    return dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), mean=float(e.mean()), n=int(len(e)))


def mmap_images(path):
    """Map the uncompressed imgs.npy member without copying/CRC-scanning the ZIP stream."""
    import zipfile, struct
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("imgs.npy")
        assert info.compress_type == zipfile.ZIP_STORED
        with archive.open(info) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(version)
            payload = stream.tell()
        with open(path, "rb") as raw:
            raw.seek(info.header_offset)
            header = raw.read(30)
            assert header[:4] == b"PK\x03\x04"
            name_len, extra_len = struct.unpack("<HH", header[26:30])
        offset = info.header_offset + 30 + name_len + extra_len + payload
        assert dtype == np.uint8 and not fortran and shape[1:] == (224, 224, 3)
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)


def load_rows(keys, stride=1, inimg=False):
    """Concatenate cached episodes. Row mask (labels inside the image if inimg), then a GLOBAL [::stride] over the masked rows
    (as gaze_lab train_gt3d: np.nonzero(ok)[0][::stride]). Returns imgs uint8 (N,224,224,3), uv float32 (N,2), episode id
    (N,) int (index into keys), row (N,) int."""
    metas = []
    for k in keys:
        with np.load(ep_file(k)) as z:
            metas.append((z["uv"].astype(np.float32), z["rows"].astype(np.int64), z["inimg"].astype(bool)))
    uv = np.concatenate([m[0] for m in metas]) if metas else np.zeros((0, 2), np.float32)
    row = np.concatenate([m[1] for m in metas]) if metas else np.zeros(0, np.int64)
    ok = np.concatenate([m[2] if inimg else np.ones(len(m[0]), bool) for m in metas]) if metas else np.zeros(0, bool)
    epi = np.concatenate([np.full(len(m[0]), i) for i, m in enumerate(metas)]) if metas else np.zeros(0, np.int64)
    sel = np.nonzero(ok)[0][::stride]
    imgs = np.empty((len(sel), IMS, IMS, 3), np.uint8)
    start, w = 0, 0
    for i, k in enumerate(keys):
        n = len(metas[i][0])
        loc = sel[(sel >= start) & (sel < start + n)] - start
        if len(loc):
            imgs[w:w + len(loc)] = mmap_images(ep_file(k))[loc]
            w += len(loc)
        start += n
    assert w == len(sel)
    return imgs, uv[sel], epi[sel], row[sel]


class AugDS(torch.utils.data.Dataset):
    """In-RAM uint8 frames + pixel labels. Train augmentation = the robust recipe (gaze_lab train_gt3d.DS with
    --bright 0.4 --chan 0.15 --blur 2.0): horizontal flip p 0.5 (u -> 223 - u), brightness x U(0.6, 1.4), per-channel
    x U(0.85, 1.15), Gaussian blur sigma U(0.5, 2) px with p 0.4, clip to [0, 1]."""

    def __init__(self, imgs, uv, train, aug=None):
        self.imgs, self.uv, self.train = imgs, uv, train
        self.aug = aug or dict(flip=True, bright=0.4, chan=0.15, blur=2.0)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, j):
        im = np.asarray(self.imgs[j], dtype=np.float32) / 255.0
        uv = self.uv[j].astype(np.float32).copy()
        if self.train:
            a = self.aug
            if a["flip"] and random.random() < 0.5:
                im = im[:, ::-1].copy(); uv[0] = IMS - 1 - uv[0]
            if a["bright"] > 0:
                im = im * (1 - a["bright"] + 2 * a["bright"] * random.random())
            if a["chan"] > 0:
                im = im * (1 - a["chan"] + 2 * a["chan"] * np.random.rand(1, 1, 3)).astype(np.float32)
            if a["blur"] > 0 and random.random() < 0.4:
                import cv2
                im = cv2.GaussianBlur(np.ascontiguousarray(im), (0, 0), random.uniform(0.5, a["blur"]))
            im = np.clip(im, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(im, dtype=np.float32)).permute(2, 0, 1), torch.from_numpy(uv)


def worker_init(_):
    import cv2
    cv2.setNumThreads(1)


def load_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_name("." + p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)
