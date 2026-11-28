"""Train one member of the cart gaze ensemble (2026-10-27 copy of the fridge one, unchanged code).

Recipe = the deployed "robust" members (gaze_lab runs/r01*_fix_heat7_blur, src/train_gt3d.py): heat7 from ImageNet (not from the basket
weights), smooth-L1 on pixels, AdamW lr 3e-4 / wd 1e-4, cosine over 30 epochs stepped per epoch, batch 64 (drop_last), bf16 autocast,
grad-norm clip 5, the 60 Hz rows at stride 2, train labels inside the image only (as its "fix" target mask), augmentation flip +
brightness 0.4 + per-channel 0.15 + Gaussian blur up to 2 px (p 0.4); input RGB/255 without normalisation (= HeatGaze.predict_uv).
Only the data differ: cart frames with MPS-gaze labels (common.py).
Checkpoint selection: best median angular error on the SELECTION episodes (a disjoint 10 % of the training episodes, as gaze_lab
train_multiscene.py after its selection-on-test fix). The held-out test episodes are logged every epoch but never used to select.
Requeue-safe: model / optimiser / scheduler / RNG state saved after every epoch (OUT/ckpt.pt) and resumed.

usage: python train.py --split splits/split_final.json --seed 0 --out runs/final_s0
writes OUT/{args.json, log.jsonl, best.pt (state_dict, CPU tensors), last.pt, ckpt.pt, summary.json}
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import AugDS, Heat7, ang_err_t, load_json, load_rows, stats, worker_init, write_json  # noqa: E402


def cpu_sd(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def save_atomic(obj, path):
    import shutil
    assert shutil.disk_usage(path.parent).free > 10*2**30, "Storage floor: preserve ongoing policy training"
    tmp = path.with_name("." + path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def evaluate(net, loader):
    net.eval()
    e = []
    with torch.inference_mode():
        for x, y in loader:
            e.append(ang_err_t(net(x.cuda(non_blocking=True)).float(), y.cuda(non_blocking=True)).cpu())
    return torch.cat(e).numpy()


def main():
    # Multiprocessing socket cleanup must use local disk, not NFS .nfs files.
    import tempfile
    local = Path("/tmp") / ("cxin34-cloth-new2-gaze-" + os.environ.get("SLURM_JOB_ID", str(os.getpid())))
    local.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert local.stat().st_uid == os.getuid()
    os.environ["TMPDIR"] = str(local)
    tempfile.tempdir = str(local)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--bright", type=float, default=0.4)
    ap.add_argument("--chan", type=float, default=0.15)
    ap.add_argument("--blur", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stop-after", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"[{out.name}] already finished ({out / 'summary.json'})", flush=True)
        return
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8, "needs an Ampere+ GPU (the recipe trains under bf16 autocast)"
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    if not (out / "args.json").exists():
        write_json(out / "args.json", vars(a))
    S = load_json(a.split)
    t0 = time.time()
    tr_imgs, tr_uv, _, _ = load_rows(S["train"], stride=a.stride, inimg=True)
    se_imgs, se_uv, _, _ = load_rows(S["selection"], stride=a.stride, inimg=True)
    te_imgs, te_uv, _, _ = load_rows(S["test"], stride=a.stride, inimg=False)
    print(f"[{out.name}] split {S['name']} seed {a.seed}: train {len(tr_uv)} frames ({len(S['train'])} eps), selection {len(se_uv)} "
          f"({len(S['selection'])} eps), test-log {len(te_uv)} ({len(S['test'])} eps), stride {a.stride}; loaded in {time.time() - t0:.0f} s "
          f"on {torch.cuda.get_device_name(0)}", flush=True)
    aug = dict(flip=True, bright=a.bright, chan=a.chan, blur=a.blur)
    tr = torch.utils.data.DataLoader(AugDS(tr_imgs, tr_uv, True, aug), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                     drop_last=True, persistent_workers=True, pin_memory=True, worker_init_fn=worker_init)
    # evaluation loaders run in-process (frames already in RAM, no augmentation): per-epoch worker processes would each leave an
    # NFS 'resource busy' traceback when multiprocessing cleans its temp dir under TMPDIR
    sl = torch.utils.data.DataLoader(AugDS(se_imgs, se_uv, False), batch_size=128, num_workers=0, pin_memory=True)
    te = torch.utils.data.DataLoader(AugDS(te_imgs, te_uv, False), batch_size=128, num_workers=0, pin_memory=True)

    net = Heat7(pretrained=True).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    start, best, best_ep, best_sd = 0, 1e9, -1, None
    ck = out / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, best, best_ep, best_sd = st["epoch"] + 1, st["best"], st["best_ep"], st["best_sd"]
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state(st["cuda_rng"])
        np.random.set_state(st["np_rng"]); random.setstate(st["py_rng"])
        write_json(out / "resume-verified.json", {"from_epoch": st["epoch"], "optimizer_restored": True, "scheduler_restored": True, "rng_restored": True})
        print(f"[{out.name}] RESUMED after epoch {st['epoch']} (best selection median {best:.2f} at epoch {best_ep})", flush=True)
    log = open(out / "log.jsonl", "a")
    for ep in range(start, a.epochs):
        net.train()
        tl, te0 = [], time.time()
        for x, y in tr:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                uvp = net(x)
            loss = F.smooth_l1_loss(uvp.float(), y)
            opt.zero_grad(set_to_none=True)
            assert torch.isfinite(loss), "Nonfinite loss"
            loss.backward()
            gradnorm=torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            assert torch.isfinite(gradnorm), "Nonfinite gradient"
            opt.step()
            tl.append(float(loss))
        sched.step()
        es, et = evaluate(net, sl), evaluate(net, te)
        rec = {"ep": ep, "train_loss": float(np.mean(tl)), **{f"sel_{k}": v for k, v in stats(es).items()},
               **{f"test_{k}": v for k, v in stats(et).items()}, "epoch_s": round(time.time() - te0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if rec["sel_median"] < best:
            best, best_ep, best_sd = rec["sel_median"], ep, cpu_sd(net)
        save_atomic(dict(epoch=ep, net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), best=best, best_ep=best_ep,
                         best_sd=best_sd,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), np_rng=np.random.get_state(),
                         py_rng=random.getstate()), ck)
        print(f"[{out.name}] ep{ep:3d} loss {rec['train_loss']:.3f} | sel med {rec['sel_median']:.2f} p90 {rec['sel_p90']:.2f} | "
              f"test(log) med {rec['test_median']:.2f} p90 {rec['test_p90']:.2f} | best sel {best:.2f} @ep{best_ep} | {rec['epoch_s']} s",
              flush=True)
        if a.stop_after and ep + 1 >= a.stop_after:
            return
    save_atomic(best_sd, out / "best.pt")             # written once, from the state the checkpoint carried (resume-consistent)
    save_atomic(cpu_sd(net), out / "last.pt")
    write_json(out / "summary.json", dict(exp=out.name, split=S["name"], seed=a.seed, best_sel_median_deg=best, best_epoch=best_ep,
                                          n_train=int(len(tr_uv)), n_sel=int(len(se_uv)), n_test_log=int(len(te_uv)), args=vars(a)))
    print(f"[{out.name}] DONE: best selection median {best:.2f} deg at epoch {best_ep} -> {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
