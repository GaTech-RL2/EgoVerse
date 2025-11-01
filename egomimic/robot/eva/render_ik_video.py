"""
Wrapper script to render IK videos with proper headless rendering setup.
Must be run BEFORE importing mujoco.
"""

import os
import sys

# IMPORTANT: Set rendering backend BEFORE importing mujoco
os.environ['MUJOCO_GL'] = 'osmesa'  # Use osmesa for headless rendering

# Now we can import everything else
import numpy as np
from pathlib import Path
import mujoco
import argparse
from scipy.spatial.transform import Rotation as R
import imageio
from tqdm import tqdm


def interpolate_trajectory(start_joints, end_joints, num_steps=50):
    """Create smooth interpolation between joint configurations."""
    trajectory = np.linspace(start_joints, end_joints, num_steps)
    return trajectory


def render_trajectory_headless(
    model,
    trajectory,
    width=640,
    height=480,
    camera_distance=1.5,
    camera_azimuth=0,
    camera_elevation=-10,
):
    """
    Render a trajectory using offscreen rendering.
    
    Args:
        model: MuJoCo model
        trajectory: Array of joint configurations (N, num_joints)
        width: Frame width
        height: Frame height
        camera_distance: Camera distance from target
        camera_azimuth: Camera azimuth angle (degrees) - 0 is front view
        camera_elevation: Camera elevation angle (degrees)
        
    Returns:
        frames: List of RGB images
    """
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # Set up camera - position to face the front of the robot
    camera = mujoco.MjvCamera()
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth  # 0 degrees = front view
    camera.elevation = camera_elevation
    camera.lookat[:] = [0.4, 0.0, 0.7]  # Point camera at robot workspace center
    
    frames = []
    
    for joints in tqdm(trajectory, desc="Rendering"):
        # Set joint positions
        data.qpos[:len(joints)] = joints
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        # Update scene and render
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels.copy())
    
    renderer.close()
    return frames


def solve_ik_trajectory(xml_path, targets, verbose=True):
    """
    Solve IK for a list of target positions and orientations.
    
    Args:
        targets: List of tuples (name, position, rotation_matrix or None)
    
    Returns list of joint configurations.
    """
    # Import here to avoid circular import issues
    from egomimic.robot.eva_kinematics import EvaMinkKinematicsSolver
    
    solver = EvaMinkKinematicsSolver(urdf_path=str(xml_path))
    
    # Home configuration
    home_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    home_pos, home_rot = solver.fk(home_joints)
    home_rot_mat = home_rot.as_matrix()
    
    if verbose:
        print(f"Home position: {home_pos}")
        print(f"Solving IK for {len(targets)} targets...")
    
    # Solve IK for all targets
    joint_trajectory = [home_joints]
    current_joints = home_joints.copy()
    
    for i, target in enumerate(targets):
        # Unpack target (can be 2 or 3 elements)
        if len(target) == 3:
            name, target_pos, target_rot = target
        else:
            name, target_pos = target
            target_rot = home_rot_mat  # Use home rotation if not specified
        
        if verbose:
            print(f"\n  Target {i+1}/{len(targets)} ({name}): pos={target_pos}")
        
        solution = solver.ik(target_pos, target_rot, current_joints)
        
        if solution is not None:
            # Verify solution
            achieved_pos, _ = solver.fk(solution)
            error = np.linalg.norm(achieved_pos - target_pos)
            if verbose:
                print(f"    Converged (error: {error*1000:.2f} mm)")
            
            joint_trajectory.append(solution)
            current_joints = solution.copy()
        else:
            if verbose:
                print(f"    Failed to converge, skipping")
    
    # Return to home
    joint_trajectory.append(home_joints)
    
    return joint_trajectory, solver.model


def create_video(
    xml_path,
    targets,
    output_path="ik_demo.mp4",
    width=1280,
    height=720,
    fps=30,
    steps_per_segment=30,
    camera_distance=2.0,
):
    """Create a video showing robot IK movements."""
    
    print("=" * 60)
    print("IK Video Renderer (Headless)")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height} @ {fps} fps")
    print()
    
    # Solve IK
    joint_trajectory, model = solve_ik_trajectory(xml_path, targets)
    
    print(f"\n  Total waypoints: {len(joint_trajectory)}")
    
    # Create smooth interpolated trajectory
    full_trajectory = []
    for i in range(len(joint_trajectory) - 1):
        segment = interpolate_trajectory(
            joint_trajectory[i],
            joint_trajectory[i + 1],
            num_steps=steps_per_segment
        )
        full_trajectory.append(segment)
    
    full_trajectory = np.vstack(full_trajectory)
    print(f"  Total frames: {len(full_trajectory)}")
    
    # Render
    print(f"\nRendering video...")
    frames = render_trajectory_headless(
        model,
        full_trajectory,
        width=width,
        height=height,
        camera_distance=camera_distance,
    )
    
    # Save video
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"[OK] Video saved successfully!")
    
    return output_path


def create_demo_video(
    xml_path,
    output_path="ik_demo.mp4",
    width=1280,
    height=720,
):
    """Create a demo video with predefined movements."""
    
    # Load model to get home position
    from egomimic.robot.eva_kinematics import EvaMinkKinematicsSolver
    solver = EvaMinkKinematicsSolver(urdf_path=str(xml_path))
    home_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    home_pos, home_rot = solver.fk(home_joints)
    
    # Define movements: up, left, right, rotate
    movement_distance = 0.1  # 10cm movements
    home_rot_mat = home_rot.as_matrix()

    up_pos = home_pos + np.array([0.0, 0.0, movement_distance])
    
    targets = [
        ("Up", up_pos, home_rot_mat),
        ("Left", up_pos + np.array([0.0, movement_distance, 0.0]), home_rot_mat),
        ("Right", up_pos + np.array([0.0, -movement_distance, 0.0]), home_rot_mat),
    ]
    
    # Add wrist rotation movements (rotate around Z axis at home position)
    wrist_rot_angles = [30, 60, 30, 0]  # degrees - rotate in one direction then back
    for angle in wrist_rot_angles:
        # Rotate around Z axis (wrist roll)
        rot_z = R.from_euler('z', angle, degrees=True)
        new_rot = (rot_z * home_rot).as_matrix()  # Apply rotation and convert to matrix
        targets.append((f"Rotate {angle}°", up_pos, new_rot))
    
    return create_video(
        xml_path,
        targets,
        output_path=output_path,
        width=width,
        height=height,
        fps=30,
        steps_per_segment=30,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render Eva robot IK videos (headless)"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="x5_scene_mod.xml",
        help="Path to MuJoCo XML scene file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ik_demo.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.0,
        help="Camera distance from robot",
    )
    
    args = parser.parse_args()
    
    # Get absolute path
    xml_path = Path(__file__).parent / args.xml
    
    if not xml_path.exists():
        print(f"Error: Scene file not found: {xml_path}")
        return 1
    
    # Create demo video
    create_demo_video(
        xml_path,
        output_path=args.output,
        width=args.width,
        height=args.height,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

