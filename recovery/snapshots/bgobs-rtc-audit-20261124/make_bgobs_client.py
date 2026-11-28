#!/usr/bin/env python3
"""Generate rollout_rby1_policy_real_bgobs.py from rollout_rby1_policy_real.py (2026-11-20).

The original builds every RTC/async prefetch observation on the main control thread, so no
action rows are sent while it runs (~300 ms per query for the cart hd2 point cloud). The
generated copy adds --bg-obs: the prefetch observation is built in the existing inference
worker thread together with the server call, and the main loop keeps executing rows.

The obs-build block is moved verbatim into a nested function _bg_build_obs() that shares its
variables with run_rollout via `nonlocal` (same semantics as inline). The only edits inside it:
  * every `continue` (stale / missing frame, empty crop) -> `return None`;
  * `x: T = v` -> `x = v` (an annotated name can't be nonlocal);
  * in the worker: robot state and odometry come from a snapshot the main thread took at the
    submit tick; base hold/pause and the operator windows are left to the main thread.
If the worker returns None, the RTC snapshot is undone and the next tick rebuilds the obs on
the main thread with the original logic (stale-frame hold etc.).
Without --bg-obs the generated client behaves like the original.

Anchors are exact line numbers + text of the original; any drift fails an assert before
anything is written, so re-check the anchors before regenerating after the original changes.

    python utils/make_bgobs_client.py --src <rollout_rby1_policy_real.py> --out <..._bgobs.py>
"""
import argparse
import ast
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(_HERE, "rollout_rby1_policy_real.py"))
ap.add_argument("--out", default=os.path.join(_HERE, "rollout_rby1_policy_real_bgobs.py"))
a = ap.parse_args()

text = open(a.src).read()
L = text.split("\n")


def line(n):
    return L[n - 1]


def expect(n, s):
    assert line(n).strip() == s.strip(), (n, line(n), s)


B0, B1 = 3092, 3885  # obs-build block inside the "else:" branch (indent 36)
WHILE = 2939
expect(3091, "else:"); expect(B0, "# Robot state"); expect(B1, "# breakpoint()")
expect(3886, "t0 = time.perf_counter()")
expect(3906, "infer_ms = (time.perf_counter() - t0) * 1000.0")
expect(3907, "if resp is not None:")
expect(3089, "resp, infer_ms = _fut.result()"); expect(3090, "async_pending = None")
expect(WHILE, "while (")
expect(2205, "aria_image_reader = AriaImageReader()")
expect(3097, "RBY1.ctrl.set_motion_paused(True)")
expect(3098, "robot_state = RBY1.robot.get_state()")
expect(3176, "RBY1.ctrl.hold()"); expect(3361, "RBY1.ctrl.hold()")
expect(3737, "odom=RBY1.get_current_odometry() if RBY1 is not None else None,")
expect(3845, "if args.show_camera and not (")
expect(3218, "t_slam_read=time.time())")
expect(3219, "percep_out = percep.compute_depth(pair[0], pair[1])")
expect(3433, "_raw = read_aria_raw_rgb(aria_streaming_shared)")
expect(71, "import traceback")
expect(4670, '"--fast-obs",'); expect(4675, ")")

tree = ast.parse(text)
fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_rollout"][0]
cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AriaImageReader"][0]
CLS_END = cls.end_lineno

# `continue`s in the block: all must target the main while-loop
conts = []


def _walk(n, loops):
    for c in ast.iter_child_nodes(n):
        if isinstance(c, (ast.For, ast.While)):
            _walk(c, loops + [c.lineno])
        elif isinstance(c, (ast.FunctionDef, ast.Lambda, ast.AsyncFunctionDef)):
            _walk(c, [])
        else:
            if isinstance(c, ast.Continue) and B0 <= c.lineno <= B1:
                assert loops and loops[-1] == WHILE, c.lineno
                conts.append(c.lineno)
            _walk(c, loops)


_walk(tree, [])
assert len(conts) == 8, conts
for c in conts:
    assert line(c).strip() == "continue", c
assert not [n for n in ast.walk(fn)
            if isinstance(n, (ast.Return, ast.Break, ast.Yield)) and B0 <= n.lineno <= B1]
assert not [n for n in ast.walk(fn)
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)) and B0 <= n.lineno <= B1]

# annotated assignments can't be nonlocal: rewrite `x: T = v` -> `x = v` (all single-line)
ann = {}
for n in ast.walk(fn):
    if isinstance(n, ast.AnnAssign) and B0 <= n.lineno <= B1:
        assert n.lineno == n.end_lineno and isinstance(n.target, ast.Name) and n.value is not None, n.lineno
        s = line(n.lineno)
        ann[n.lineno] = f"{s[: len(s) - len(s.lstrip())]}{n.target.id} = {ast.get_source_segment(text, n.value)}"

# names the block binds -> nonlocal (shared with run_rollout exactly as inline); "_" stays local
comp = set()
for n in ast.walk(fn):
    if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        for g in n.generators:
            for m in ast.walk(g.target):
                if isinstance(m, ast.Name):
                    comp.add((m.id, m.lineno, m.col_offset))


def stores(lo, hi, excl=None):
    s = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and lo <= n.lineno <= hi:
            if excl and excl[0] <= n.lineno <= excl[1]:
                continue
            if (n.id, n.lineno, n.col_offset) in comp:
                continue
            s.add(n.id)
    return s


blk = stores(B0, B1) - {"_"}
needs_init = sorted(blk - stores(fn.lineno, fn.end_lineno, (B0, B1)))
nonlocals = sorted(blk)
assert "step" not in blk and "prefetch_now" not in blk

I20, I36, I40 = " " * 20, " " * 36, " " * 40
EDITS = {
    3097: ("RBY1.ctrl.set_motion_paused(True)", "if not _in_bg: RBY1.ctrl.set_motion_paused(True)"),
    3098: ("RBY1.robot.get_state()", "(_pre_state if _pre_state is not None else RBY1.robot.get_state())"),
    3176: ("RBY1.ctrl.hold()", "_bg_hold_req.append(1) if _in_bg else RBY1.ctrl.hold()"),
    3361: ("RBY1.ctrl.hold()", "_bg_hold_req.append(1) if _in_bg else RBY1.ctrl.hold()"),
    3737: ("odom=RBY1.get_current_odometry() if RBY1 is not None else None,",
           "odom=(_pre_odom if _in_bg else (RBY1.get_current_odometry() if RBY1 is not None else None)),"),
    3845: ("if args.show_camera and not (", "if (not _in_bg) and args.show_camera and not ("),
    # bg-obs colour/depth alignment (2026-11-20): use the RAW RGB frame read together with the
    # SLAM pair (below, before the depth compute) instead of one read ~100 ms later.
    3433: ("_raw = read_aria_raw_rgb(aria_streaming_shared)",
           "_raw = _raw_pre if _raw_pre is not None else read_aria_raw_rgb(aria_streaming_shared)"),
}
# inserted BEFORE these block lines (same indentation as that line)
INSERT_BEFORE = {
    3219: ["if _in_bg:",
           "    # bg-obs: the robot keeps moving during the build, so read the colour frame",
           "    # together with the stereo pair, not after the ~100 ms depth compute.",
           "    _raw_pre = read_aria_raw_rgb(aria_streaming_shared)"],
}

out = [
    "# rollout_rby1_policy_real_bgobs.py — GENERATED by utils/make_bgobs_client.py from rollout_rby1_policy_real.py",
    "# (working tree, 2026-11-20). Only change: --bg-obs builds the RTC/async prefetch observation in the worker",
    "# thread with the inference, so rows keep executing meanwhile. Without --bg-obs it behaves like the original.",
]
for n in range(1, len(L) + 1):
    s = line(n)
    if n == 71:
        out += [s, "import threading"]
        continue
    if n == CLS_END:
        out += [s, "", "",
                "class _LockedAriaImageReader(AriaImageReader):",
                '    """--bg-obs: the obs worker and the main loop\'s camera window share one reader.',
                "",
                "    Each shared-memory read is serialized (read_fresh loops over read(), so it is locked",
                '    per read, never for its whole wait)."""',
                "",
                "    def __init__(self) -> None:",
                "        super().__init__()",
                "        self._bg_lock = threading.Lock()",
                "",
                "    def read(self, aria_shared):",
                "        with self._bg_lock:",
                "            return super().read(aria_shared)"]
        continue
    if n == 2205:
        out += [s,
                '    if getattr(args, "bg_obs", False):',
                "        aria_image_reader = _LockedAriaImageReader()"]
        continue
    if n == WHILE:
        out += [I20 + "# ---- --bg-obs: build the prefetch obs in the worker thread (generated) ----",
                I20 + "_bg_hold_req: list = []   # worker asks the main thread to re-anchor the base (stale frame)",
                I20 + "_bg_force_sync = False    # next prefetch: build on the main thread (original path)",
                I20 + "_bg_inflight = False      # the pending future is a bg (obs + infer) job",
                I20 + "_bg_rtc_saved = None",
                I20 + "bg_obs_on = False",
                I20 + 'if getattr(args, "bg_obs", False):',
                I20 + "    _bg_why = []",
                I20 + "    if async_pool is None:",
                I20 + '        _bg_why.append("needs --rtc or --async-infer")',
                I20 + "    if (args.use_gt_action or args.use_gt_proprio or args.gt_image",
                I20 + "            or args.gt_proprio_live_image or args.zero_obs):",
                I20 + '        _bg_why.append("live obs only (no GT / zero-obs modes)")',
                I20 + "    if base_pose_history is not None or progress_mem is not None or state_hist is not None or v2_live:",
                I20 + '        _bg_why.append("base-pose history / progress / state history / v2 obs are main-thread state")',
                I20 + "    if args.record_image_compare_dir:",
                I20 + '        _bg_why.append("--record-image-compare-dir")',
                I20 + "    bg_obs_on = not _bg_why",
                I20 + "    if bg_obs_on:",
                I20 + "        print(colored(",
                I20 + '            "[bg-obs] ON: prefetch observations are built in the worker thread; rows keep "',
                I20 + '            "executing meanwhile. RTC delta now also counts the obs build: keep "',
                I20 + '            "--rtc-freeze-steps >= the printed delta (LATE lines mean raise it).", "cyan"))',
                I20 + "    else:",
                I20 + '        print(colored("[bg-obs] OFF (" + "; ".join(_bg_why) + "): obs built on the main thread", "yellow"))']
        for i in range(0, len(needs_init), 8):
            out.append(I20 + " = ".join(needs_init[i:i + 8]) + " = None")
        out += [I20 + "def _bg_show_windows(_p, _pv):",
                I20 + '    """Main thread: the operator windows the inline obs build would have drawn."""',
                I20 + "    if args.show_camera and not (args.use_gt_proprio or args.gt_image or args.gt_proprio_live_image):",
                I20 + '        if args.obs_mode != "rgb" and _pv is not None:',
                I20 + "            show_perception_window(_pv)",
                I20 + "        else:",
                I20 + '            _cam = _p.get("front_img_1")',
                I20 + "            if _cam is None:",
                I20 + "                _cam = last_aria_frame",
                I20 + "            if _cam is not None:",
                I20 + "                show_policy_camera_window(_cam, args.policy_image_size)",
                s]
        continue
    if n == 3089:
        ind = s[: len(s) - len(s.lstrip())]
        out += [ind + "_bg_out = _fut.result()",
                ind + "resp, infer_ms = _bg_out[0], _bg_out[1]"]
        continue
    if n == 3090:
        ind = s[: len(s) - len(s.lstrip())]
        out += [s,
                ind + "if _bg_inflight:",
                ind + "    _bg_inflight = False",
                ind + "    while _bg_hold_req:",
                ind + "        _bg_hold_req.pop()",
                ind + "        try:",
                ind + "            RBY1.ctrl.hold()",
                ind + "        except Exception as _hold_exc:",
                ind + '            print(colored(f"[hold] base re-anchor failed: {_hold_exc}", "yellow"))',
                ind + "    if resp is None:",
                ind + "        # The worker could not build this obs (stale / missing frame, empty crop):",
                ind + "        # undo the RTC snapshot; next tick rebuilds it on the main thread with the",
                ind + "        # original logic (stale-frame hold, waits).",
                ind + "        if rtc is not None:",
                ind + "            rtc.pending = False",
                ind + "            rtc._last_snapshot_step = _bg_rtc_saved",
                ind + "        _bg_force_sync = True",
                ind + "    else:",
                ind + "        _bg_p, _bg_pv, _bg_obs_ms = _bg_out[2]",
                ind + "        _bg_show_windows(_bg_p, _bg_pv)",
                ind + "        if args.print_every > 0:",
                ind + '            print(f"[bg-obs] obs {_bg_obs_ms:.0f} ms + infer {infer_ms:.0f} ms (rows kept running)")']
        continue
    if n == B0:
        out += [I36 + "def _bg_build_obs(step, prefetch_now, _pre_state, _pre_odom, _in_bg):",
                I40 + '"""The inline obs build, moved verbatim (see utils/make_bgobs_client.py). Returns',
                I40 + '(payload, perception preview), or None where the inline code did `continue`."""']
        for i in range(0, len(nonlocals), 8):
            out.append(I40 + "nonlocal " + ", ".join(nonlocals[i:i + 8]))
        out.append(I40 + "_raw_pre = None   # colour frame read with the SLAM pair (bg-obs only)")
    if B0 <= n <= B1:
        if n in INSERT_BEFORE:
            ind = "    " + s[: len(s) - len(s.lstrip())]
            out += [ind + x for x in INSERT_BEFORE[n]]
        t = ann.get(n, s)
        if n in conts:
            t = s.replace("continue", "return None")
        elif n in EDITS:
            x, y = EDITS[n]
            assert s.count(x) == 1, n
            t = s.replace(x, y)
        out.append(("    " + t) if t.strip() else t)
        if n == B1:
            out += [I40 + "return payload, percep_preview",
                    I36 + "if bg_obs_on and prefetch_now and not _bg_force_sync:",
                    I40 + "# --bg-obs: snapshot robot state + the RTC tail NOW; build the obs and infer",
                    I40 + "# in the worker while this loop keeps executing the current chunk.",
                    I40 + "_bg_pre_state = RBY1.robot.get_state() if RBY1 is not None else None",
                    I40 + "_bg_pre_odom = RBY1.get_current_odometry() if RBY1 is not None else None",
                    I40 + "_bg_rtc_keys = {}",
                    I40 + "_bg_rtc_saved = rtc._last_snapshot_step if rtc is not None else None",
                    I40 + "if rtc is not None:",
                    I40 + "    rtc.attach(_bg_rtc_keys, step)",
                    I40 + "def _bg_worker(_s=step, _st=_bg_pre_state, _od=_bg_pre_odom, _rk=_bg_rtc_keys):",
                    I40 + "    _tb = time.perf_counter()",
                    I40 + "    _res = _bg_build_obs(_s, True, _st, _od, True)",
                    I40 + "    if _res is None:",
                    I40 + "        return None, -1.0, None",
                    I40 + "    _p, _pv = _res",
                    I40 + "    if args.viz_num_samples and args.viz_num_samples > 1:",
                    I40 + '        _p["num_samples"] = int(args.viz_num_samples)',
                    I40 + "    _p.update(_rk)",
                    I40 + "    _obs_ms = (time.perf_counter() - _tb) * 1000.0",
                    I40 + "    _t = time.perf_counter()",
                    I40 + "    _r = client.infer(_p)",
                    I40 + "    return _r, (time.perf_counter() - _t) * 1000.0, (_p, _pv, _obs_ms)",
                    I40 + "async_pending = (async_pool.submit(_bg_worker), step, time.perf_counter())",
                    I40 + "_bg_inflight = True",
                    I36 + "else:",
                    I40 + "if _bg_build_obs(step, prefetch_now, None, None, False) is None:",
                    I40 + "    # stale-frame hold / waits: stay on the main thread (no rows run, as",
                    I40 + "    # in the original) until an obs builds again, then go back to --bg-obs",
                    I40 + "    continue",
                    I40 + "_bg_force_sync = False"]
        continue
    if 3886 <= n <= 3906:
        out.append(("    " + s) if s.strip() else s)
        continue
    if n == 4675:
        out += [s,
                "    parser.add_argument(",
                '        "--bg-obs",',
                '        action="store_true",',
                '        help="(rollout_rby1_policy_real_bgobs.py) Build the RTC/async prefetch observation in the worker "',
                '             "thread too, so rows keep executing while it is built (no obs-build pause). RTC delta then "',
                '             "also counts the obs time: keep --rtc-freeze-steps >= the printed delta.",',
                "    )"]
        continue
    out.append(s)

o = "\n".join(out)
compile(o, a.out, "exec")  # fail before writing anything
with open(a.out, "w") as f:
    f.write(o)
print(f"wrote {a.out}: {len(out)} lines; {len(nonlocals)} shared names, {len(needs_init)} pre-bound")
