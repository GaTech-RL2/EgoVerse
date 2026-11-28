"""11-02 hand-dense global cloud (user 11-02 ~23:05: "FPS turns out to be very harmful ... less than 10 points on the handle are kept,
especially when both hands are occluding the handles ... for that 2048 points ... a 10 or 15 cm ball [around the hands] keeps all the
possible points ... one fourth, 512 points around the hands, and the other points scattered around").
Layout of the 2048 points (fixed segments, read by the ptf encoder's segment-aware grouping 16 / 16 / 32 patches):
  [0, 512)     right-hand segment: points within HR (0.15 m) of the right eef (nearer eef wins), depth slab only (no lateral / gaze
               crop), FPS to 512 if more; fewer than 512 -> all of them, repeated at random to 512; fewer than MINH (64) -> those
               points + scene points (the hand is out of view)
  [512, 1024)  left-hand segment, the same
  [1024, 2048) the rest: FPS over the global crop (A4 lateral + depth + gaze ball) minus both hand balls
Colours: as given (the builder greys everything within 0.15 m of either eef before this, so the hand segments are geometry only).
The same function serves the training build and the robot client (vendor this file)."""
import torch
from egomimic.models.adapt3r_3d_encoder import fps_pytorch

NG, NH, HR, MINH, CAP = 2048, 512, 0.15, 64, 16384


def _fps(P, n):
    """indices of an FPS-ordered subset of n points (P has >= n points)"""
    return fps_pytorch(P[None], n)[0] if P.shape[0] > n else torch.arange(P.shape[0], device=P.device)


def _cap(idx, gen, n=CAP):
    if idx.numel() > n:
        idx = idx[torch.randperm(idx.numel(), generator=gen, device=idx.device)[:n]]
    return idx


def hand_dense_global(Pg, C, peR, peL, slab, mrest, gen, ng=NG, nh=NH, hr=HR, minh=MINH):
    """Pg (N, 3) robot-glass points, C (N, 3) colours, peR / peL (3,) eef positions in glass, slab (N,) depth-slab mask,
    mrest (N,) global-crop mask, gen = torch.Generator on Pg's device. Returns (ng, 6) xyzrgb and the two hand-pool sizes."""
    dR, dL = torch.linalg.norm(Pg - peR, dim=1), torch.linalg.norm(Pg - peL, dim=1)
    inR, inL = slab & (dR < hr) & (dR <= dL), slab & (dL < hr) & (dL < dR)
    segs, short, sizes = [], [], []
    for m in (inR, inL):
        idx = _cap(torch.nonzero(m).squeeze(1), gen); sizes.append(int(idx.numel()))
        if idx.numel() >= minh:
            sel = idx[_fps(Pg[idx], nh)]
            if sel.numel() < nh:
                sel = torch.cat([sel, sel[torch.randint(0, sel.numel(), (nh - sel.numel(),), generator=gen, device=sel.device)]])
            segs.append(sel); short.append(0)
        else:
            segs.append(idx); short.append(nh - int(idx.numel()))
    rest = _cap(torch.nonzero(mrest & ~inR & ~inL).squeeze(1), gen)
    need = (ng - 2 * nh) + sum(short)
    if rest.numel() == 0:
        rest = _cap(torch.nonzero(slab).squeeze(1), gen)
    if rest.numel() == 0:
        out = torch.zeros(ng, 6, device=Pg.device); out[:, 3:] = 0.5
        return out, sizes
    rsel = rest[_fps(Pg[rest], need)]
    if rsel.numel() < need:
        rsel = torch.cat([rsel, rsel[torch.randint(0, rsel.numel(), (need - rsel.numel(),), generator=gen, device=rsel.device)]])
    main, extra = rsel[: ng - 2 * nh], rsel[ng - 2 * nh:]
    k = 0
    for i in range(2):                                  # top up a short hand segment with scene points
        if short[i]:
            segs[i] = torch.cat([segs[i], extra[k:k + short[i]]]); k += short[i]
    sel = torch.cat([segs[0], segs[1], main])
    assert sel.numel() == ng, sel.numel()
    return torch.cat([Pg[sel], C[sel]], 1), sizes


def hand_dense_local(Pg, C, ptcp, mball, gen, nl=1024, nh=512, hr=0.30, minh=MINH):
    """11-03 hand-dense LOCAL cloud (user 11-03 15:3x: "your local point cloud does not sample around the hands ... for local one, it's
    more like needed for sample around the hands"; half at its own hand). Pg (N, 3) robot-glass points, C (N, 3) colours, ptcp (3,) this
    cloud's own TCP in glass, mball (N,) the local ball mask (around its eef), gen = torch.Generator on Pg's device.
    Layout of the nl points: [0, nh) = points within hr of the TCP (no depth slab, like the local stream), FPS to nh, fewer -> repeated at
    random, fewer than minh -> those points + rest points; [nh, nl) = FPS over the ball minus the hand region.
    Returns (nl, 6) xyzrgb in glass and the hand-pool size."""
    inH = torch.linalg.norm(Pg - ptcp, dim=1) < hr
    idx = _cap(torch.nonzero(inH).squeeze(1), gen); size = int(idx.numel())
    if idx.numel() >= minh:
        seg = idx[_fps(Pg[idx], nh)]
        if seg.numel() < nh:
            seg = torch.cat([seg, seg[torch.randint(0, seg.numel(), (nh - seg.numel(),), generator=gen, device=seg.device)]])
        short = 0
    else:
        seg, short = idx, nh - int(idx.numel())
    rest = _cap(torch.nonzero(mball & ~inH).squeeze(1), gen)
    need = (nl - nh) + short
    if rest.numel() == 0:
        rest = _cap(torch.nonzero(mball).squeeze(1), gen)
    if rest.numel() == 0:
        out = torch.zeros(nl, 6, device=Pg.device); out[:, 3:] = 0.5
        return out, size
    rsel = rest[_fps(Pg[rest], need)]
    if rsel.numel() < need:
        rsel = torch.cat([rsel, rsel[torch.randint(0, rsel.numel(), (need - rsel.numel(),), generator=gen, device=rsel.device)]])
    sel = torch.cat([seg, rsel[nl - nh:], rsel[: nl - nh]]) if short else torch.cat([seg, rsel])
    assert sel.numel() == nl, sel.numel()
    return torch.cat([Pg[sel], C[sel]], 1), size
