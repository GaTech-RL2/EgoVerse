"""
Convert body-pose CSVs to robot HDF5 using SEWSolver with the user's __init__ kwargs.

Mirrors `convert_human_csv_to_robot_hdf5_mink.py` from the SEW-Geometric-Teleop repo:
reuses every helper from the upstream SEW converter, only the solver construction
is overridden so the user-specified flags take effect.

Run from the SEW-Geometric-Teleop checkout (so `projects.*` imports resolve):
    cd /coc/flash7/zhenyang/SEW-Geometric-Teleop
    PYTHONPATH="$PWD:$PYTHONPATH" python \
        /coc/flash7/zhenyang/EgoVerse/egomimic/scripts/sew_process/csv_to_hdf5_sew_custom.py \
        --input_folder <csv_dir> --output_folder <out_dir> --hdf5_name robot_data_sew.hdf5
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

os.environ.setdefault("PYNPUT_BACKEND", "dummy")
os.environ.setdefault("PYNPUT_BACKEND_KEYBOARD", "dummy")
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import h5py
import mujoco
from tqdm import tqdm

from projects.xhand_teleop import convert_human_csv_to_robot_hdf5 as sew_convert
from projects.xhand_teleop.controllers.rby1_with_xhand_mujoco_controller_v3 import (
    RBY1WithXHandMuJoCoController,
)
from projects.xhand_teleop.rby1_with_xhand_sew_solver import RBY1WithXHandSEWSolver

TASK_ID_ONE_HOT_DIM = sew_convert.TASK_ID_ONE_HOT_DIM
VIDEO_HEIGHT = sew_convert.VIDEO_HEIGHT
VIDEO_WIDTH = sew_convert.VIDEO_WIDTH
VIDEO_FPS = sew_convert.VIDEO_FPS

DEFAULT_MOBILE_BASE_CONFIG = "/coc/flash7/zhenyang/EgoVerse/configs/sew_solver_mobile_base_user.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Convert body-pose CSVs to robot HDF5 using SEWSolver with user-specified kwargs."
    )
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--hdf5_name", type=str, default="robot_data_sew.hdf5")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--per_demo_video", action="store_true")
    parser.add_argument("--csv_reading_freq", type=float, default=sew_convert.CSV_READING_FREQUENCY)
    parser.add_argument("--no_filter", action="store_true")
    parser.add_argument("--no_aria_black_image", action="store_true")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--mobile_base_config_path", type=str, default=DEFAULT_MOBILE_BASE_CONFIG,
                        help="YAML override for mobile_base_* / stability params.")
    args = parser.parse_args()
    sew_convert.CSV_READING_FREQUENCY = args.csv_reading_freq

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        print(f"Error: input folder not found: {input_folder}")
        return

    csv_files = sorted(glob.glob(str(input_folder / "*.csv")))
    if not csv_files:
        print(f"Error: no CSV files found in {input_folder}")
        return
    print(f"Found {len(csv_files)} CSV file(s) in {input_folder}")

    output_folder = Path(args.output_folder) if args.output_folder else Path(str(input_folder) + "_converted_sew")
    output_folder.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_folder / args.hdf5_name
    print(f"Output HDF5: {hdf5_path}")

    xml_path = sew_convert._get_xml_path()
    print(f"Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data_mj = mujoco.MjData(model)
    model.opt.timestep = 0.005

    controller = RBY1WithXHandMuJoCoController(model, data_mj, debug=False)
    controller.setup_mocap_body("base_mocap_mover")

    # User-specified SEWSolver kwargs.
    ik_solver = RBY1WithXHandSEWSolver(
        debug=False,
        enable_collision_avoidance=False,
        enable_functional_retargeting=False,
        enable_functional_offset=True,
        base_alignement_mode="manual",
        enable_elbow_angle_low_pass=False,
        elbow_angle_lp_cutoff_freq=0.01,
        elbow_angle_lp_sampling_rate=60.0,
        enable_local_p_upper_body_low_pass=False,
        local_p_upper_body_lp_cutoff_freq=0.01,
        local_p_upper_body_lp_sampling_rate=60.0,
        mobile_base=False,
        mobile_base_config_path=args.mobile_base_config_path,
    )
    print(f"SEW IK solver initialized. mobile_base={ik_solver.mobile_base}, "
          f"collision={ik_solver.enable_collision_avoidance}, "
          f"stability_clamp={ik_solver.enable_stability_clamp}, "
          f"stability_radius={ik_solver.stability_radius}")

    needed_off_w = max(int(model.vis.global_.offwidth), VIDEO_WIDTH)
    needed_off_h = max(int(model.vis.global_.offheight), VIDEO_HEIGHT)
    model.vis.global_.offwidth = needed_off_w
    model.vis.global_.offheight = needed_off_h
    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

    writer = sew_convert.RobomimicHDF5Writer()

    long_video_writer = None
    long_video_path = None
    if not args.no_video and not args.per_demo_video:
        long_video_path = str(output_folder / "all_demos.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        long_video_writer = cv2.VideoWriter(long_video_path, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    for csv_idx, csv_file in enumerate(tqdm(csv_files, desc="CSVs", unit="file")):
        demo_key = f"demo_{csv_idx}"
        csv_name = Path(csv_file).stem
        print(f"\n[{csv_idx+1}/{len(csv_files)}] Processing: {csv_name} -> {demo_key}")

        per_demo_writer = None
        per_demo_path = None
        if not args.no_video and args.per_demo_video:
            per_demo_path = str(output_folder / f"{demo_key}_{csv_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            per_demo_writer = cv2.VideoWriter(per_demo_path, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

        t0 = time.time()
        episode = sew_convert.process_single_csv(
            csv_file,
            model,
            data_mj,
            controller,
            ik_solver,
            renderer,
            video_writer=per_demo_writer if per_demo_writer is not None else long_video_writer,
            use_filter=not args.no_filter,
            use_black_aria_image=not args.no_aria_black_image,
            enable_base_motion=False,
        )
        elapsed = time.time() - t0

        if per_demo_writer is not None:
            per_demo_writer.release()

        if episode is None:
            print(f"  Skipped (no valid data)")
            continue

        plot_path = output_folder / f"{demo_key}_{csv_name}_body_apriltag_xy.png"
        sew_convert._plot_body_and_apriltag_xy(
            episode,
            plot_path,
            title=f"{demo_key} (sew-custom): body trajectory and AprilTag XY",
        )

        writer.write_episode(demo_key, episode, hdf5_path)
        if args.task_id is not None:
            with h5py.File(hdf5_path, "a") as f:
                obs_grp = f["data"][demo_key]["obs"]
                task_id_arr = sew_convert._build_task_id_one_hot(episode.num_samples, args.task_id)
                obs_grp.create_dataset("task_id", data=task_id_arr)
        print(f"  Done: {episode.num_samples} samples in {elapsed:.1f}s")

    renderer.close()
    if long_video_writer is not None:
        long_video_writer.release()
    print(f"\nAll done. SEW HDF5: {hdf5_path}")


if __name__ == "__main__":
    main()
