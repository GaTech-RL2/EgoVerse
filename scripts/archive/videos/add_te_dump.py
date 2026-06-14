import sys
p = "egomimic/eval/eval_sim.py"
src = open(p).read()
if "(ROLLOUT_VIDEO_DIR) [chunk_te]" in src:
    print("ALREADY HAS TE DUMP - skipping"); sys.exit(0)
lines = src.split("\n")
dump = (
'        # --- env-gated per-episode rollout video dump (ROLLOUT_VIDEO_DIR) [chunk_te] ---\n'
'        _rvd = os.environ.get("ROLLOUT_VIDEO_DIR")\n'
'        if _rvd and frames:\n'
'            try:\n'
'                import torchvision.io as tvio\n'
'                os.makedirs(_rvd, exist_ok=True)\n'
'                path = os.path.join(_rvd, f"roll_ep{ep_idx}_cov{last_coverage:.3f}.mp4")\n'
'                vid = torch.from_numpy(np.stack(frames, axis=0)).to(torch.uint8)\n'
'                tvio.write_video(path, vid, fps=int(self.video_fps), video_codec="h264")\n'
'                print(f"[ROLLOUT_VIDEO] ep={ep_idx} wrote {path} ({len(frames)} frames) cov={last_coverage:.3f}", flush=True)\n'
'            except Exception as _ex:\n'
'                import traceback\n'
'                print(f"[ROLLOUT_VIDEO_ERR] {_ex}", flush=True)'
)
out, i, inserted, seen = [], 0, False, False
while i < len(lines):
    line = lines[i]
    if "[HNET_CHUNKTE_DBG]" in line:
        seen = True
    if (not inserted) and seen and line.strip() == "return last_coverage, frames, actions_taken":
        out.append(dump)
        out.append(line)
        inserted = True
    else:
        out.append(line)
    i += 1
assert inserted, "ERROR: did not find chunk_te return to insert before"
open(p, "w").write("\n".join(out))
print("INSERTED chunk_te video dump OK")
