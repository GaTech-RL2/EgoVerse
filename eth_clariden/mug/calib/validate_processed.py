"""Check processed real_mug H5s against convert_faive_to_lerobot.py's parsing contract."""
import glob, sys, h5py, numpy as np

DATASET_KEY_MAPPINGS = {"qpos_arm": "cartesian_arm", "qpos_hand": "joints_hand",
                        "aria_rgb_cam/color": "aria_rgb_cam",
                        "oakd_left_wrist_view/color": "left_wrist_img",
                        "oakd_right_wrist_view/color": "right_wrist_img"}
CAMERAS_TO_IGNORE = {"oakd_front_view/color"}

def leaves(g):
    out = []
    g.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) else None)
    return out

files = sorted(glob.glob(sys.argv[1] + "/*.h5"))
print(f"{len(files)} processed files")
ok = True
for fn in files:
    with h5py.File(fn) as h:
        print("\n==", fn.split("/")[-1], dict((k, h.attrs[k]) for k in h.attrs if k in ("frequency_hz", "arm", "quat_order")))
        for n in leaves(h):
            print(f"   {n:50s} {h[n].shape} {h[n].dtype}")
        # --- emulate FaiveHD5Extractor.get_state / get_cameras / process_episode(arm='right')
        states = [p for p in leaves(h["observations"]) if "images" not in p]
        right = {}
        for s in states:
            assert "right" in s, f"state {s} has no 'right' suffix -> converter would ignore it"
            name = s.split("right")[0][:-1]
            right[DATASET_KEY_MAPPINGS.get(name, name)] = h["observations"][s][:]
        cams = [c for c in leaves(h["observations"]["images"]) if c not in CAMERAS_TO_IGNORE]
        mapped = [DATASET_KEY_MAPPINGS.get(c, c) for c in cams]
        print("   converter sees states :", sorted(right))
        print("   converter sees cameras:", cams, "->", mapped)
        assert set(right) == {"cartesian_arm", "joints_hand"}, right.keys()
        assert set(mapped) == {"aria_rgb_cam", "right_wrist_img"}, mapped
        assert "actions_arm_right" in h and "actions_hand_right" in h
        T = right["cartesian_arm"].shape[0]
        assert right["cartesian_arm"].shape == (T, 7) and right["joints_hand"].shape == (T, 17)
        assert h["actions_arm_right"].shape == (T, 7) and h["actions_hand_right"].shape == (T, 17)
        for c in cams:
            assert h["observations/images"][c].shape[0] == T and h["observations/images"][c].dtype == np.uint8
        # --- numeric sanity
        qo, qa = right["cartesian_arm"], h["actions_arm_right"][:]
        assert np.allclose(np.linalg.norm(qo[:, 3:], axis=1), 1, atol=1e-6)
        dots = np.sum(qo[:, 3:] * qa[:, 3:], axis=1)
        dp = np.linalg.norm(qo[:, :3] - qa[:, :3], axis=1) * 1000
        ang = np.degrees(2 * np.arccos(np.clip(np.abs(dots), 0, 1)))
        print(f"   |qpos_arm - actions_arm| : pos mean {dp.mean():.1f} mm max {dp.max():.1f} mm | rot mean {ang.mean():.2f} deg max {ang.max():.2f} deg | quat dot min {dots.min():.3f} (sign-continuous: {bool((dots > 0).all())})")
        hj = right["joints_hand"]; ha = h["actions_hand_right"][:]
        print(f"   hand: qpos range [{hj.min():.2f},{hj.max():.2f}] |qpos-act| mean {np.abs(hj-ha).mean():.3f} rad")
        jump = np.linalg.norm(np.diff(qo[:, :3], axis=0), axis=1).max() * 1000
        print(f"   qpos_arm max per-frame jump {jump:.1f} mm; T={T} ({T/30:.1f}s)")
        assert jump < 50, "FK proprio has discontinuities"
print("\nALL CHECKS PASSED" if ok else "FAILED")
