"""Training-input augmentation in physical metres; never touches labels/padding."""
import numpy as np
import torch

def augment(batch, config):
    out = dict(batch)
    keys = sorted(k for k in batch if k.startswith('front_pcd_'))
    arrays = {k: batch[k].detach().cpu().numpy().copy().reshape(len(batch[k]), -1, 6) for k in keys}
    for i in range(len(batch[keys[0]])):
        gray = np.random.random() < config['grayscale']['p']
        for k in keys:
            a = arrays[k][i]
            mask = np.any(a != 0, axis=-1)
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            pts = a[indices].copy()
            drop = config['point_dropout']
            if np.random.random() < drop['p']:
                lo, hi = drop['global_ratio' if k == 'front_pcd_1' else 'hand_ratio']
                keep = max(1, int(np.ceil(len(pts) * (1 - np.random.uniform(lo, hi)))))
                selected = np.random.permutation(len(pts))[:keep]
                # Every retained point survives; only the dropped slots are refilled.
                sample = np.concatenate([selected, np.random.choice(selected, len(pts)-keep)])
                np.random.shuffle(sample)
                pts = pts[sample].copy()
            jitter = config['xyz_jitter']
            if np.random.random() < jitter['p']:
                noise = np.random.normal(0, jitter['std_m'], pts[:, :3].shape)
                pts[:, :3] += np.clip(noise, -jitter['clip_m'], jitter['clip_m']).astype(pts.dtype)
            if gray:
                lum = pts[:, 3:] @ np.array([.299, .587, .114], dtype=pts.dtype)
                pts[:, 3:] = lum[:, None]
            a[indices] = pts
    for k in keys:
        out[k] = torch.from_numpy(arrays[k]).to(batch[k]).reshape_as(batch[k])
    return out
