import torch, zarr, glob, os
device = "cuda"
from egomimic.algo.vae.algo import load_pretrained_vae
vae = load_pretrained_vae("external_ckpts/vae_v5_last.ckpt").to(device).eval()

zarr_paths = sorted(glob.glob("/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle/*.zarr"))[:20]
all_mu = []
for zp in zarr_paths:
    try:
        z = zarr.open_group(zp, mode="r")
        keys = list(z.keys())
        img_key = [k for k in keys if "img" in k.lower() or "front" in k.lower()]
        if not img_key:
            print(f"  No img key in {os.path.basename(zp)}, keys={keys[:5]}")
            continue
        imgs = z[img_key[0]][:]
        imgs_t = torch.from_numpy(imgs).float().to(device)
        if imgs_t.shape[-1] == 3:
            imgs_t = imgs_t.permute(0, 3, 1, 2)
        if imgs_t.max() > 1.5:
            imgs_t = imgs_t / 255.0
        with torch.no_grad():
            mu, _ = vae.encode(imgs_t[:50])
        all_mu.append(mu.cpu())
    except Exception as e:
        print(f"  skip {os.path.basename(zp)}: {e}")

if all_mu:
    all_mu = torch.cat(all_mu, dim=0)
    print(f"Total frames: {all_mu.shape[0]}")
    for c in range(4):
        print(f"  Ch {c}: mean={all_mu[:,c].mean():.6f} std={all_mu[:,c].std():.6f}")
    print(f"  Overall: mean={all_mu.mean():.6f} std={all_mu.std():.6f}")
else:
    print("No data loaded!")
