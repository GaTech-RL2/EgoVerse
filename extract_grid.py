import sys, torch
import torchvision.io as tvio
import torchvision.utils as tvu

path = sys.argv[1]
out = sys.argv[2]
v, _, _ = tvio.read_video(path, output_format="TCHW")  # (T,C,H,W) uint8
T = v.shape[0]
n = min(8, T)
idxs = torch.linspace(0, T - 1, n).long()
frames = v[idxs].float() / 255.0
grid = tvu.make_grid(frames, nrow=4, padding=2)
tvu.save_image(grid, out)
print(f"T={T} saved {n} frames (idxs={idxs.tolist()}) -> {out}")
