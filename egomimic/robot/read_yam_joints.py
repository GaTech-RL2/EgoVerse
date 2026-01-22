#!/usr/bin/env python3
"""
Utility script to read current YAM robot joint positions without actuation.

This is useful for:
- Sanity checking robot state before teleop
- Capturing a "home" position for later use
- Verifying CAN communication is working

Usage:
    python read_yam_joints.py                      # Stream both arms
    python read_yam_joints.py --right-only         # Stream right arm only
    python read_yam_joints.py --left-only          # Stream left arm only
    python read_yam_joints.py --rate 20            # Stream at 20Hz
    python read_yam_joints.py --save-home home.npz # Save positions on exit
"""

import sys
import os
import argparse
import time
import numpy as np

# Add i2rt to path
sys.path.insert(0, os.path.expanduser("~/i2rt"))

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType


# Joint names and limits (shared between arms)
JOINT_NAMES = [
    "J1 (base)",
    "J2 (shoulder)",
    "J3 (elbow)",
    "J4 (wrist1)",
    "J5 (wrist2)",
    "J6 (wrist3)",
    "Gripper",
]

JOINT_LIMITS = [
    (-2.617, 3.13),
    (0.0, 3.65),
    (0.0, 3.13),
    (-1.57, 1.57),
    (-1.57, 1.57),
    (-2.09, 2.09),
    (0.0, 1.0),  # Gripper normalized
]


def format_joint_line(name: str, pos: float, idx: int) -> str:
    """Format a single joint position for display."""
    lo, hi = JOINT_LIMITS[idx]
    
    if idx < 6:
        deg = np.rad2deg(pos)
        # Check if near limits
        margin = 0.1
        if pos < lo + margin:
            status = "!"
        elif pos > hi - margin:
            status = "!"
        else:
            status = " "
        return f"{name}: {pos:+7.3f} rad ({deg:+7.1f}°){status}"
    else:
        # Gripper as percentage
        pct = pos * 100
        return f"{name}: {pos:+7.3f} ({pct:5.1f}%) "


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def stream_joint_positions(
    use_right: bool = True,
    use_left: bool = True,
    gripper_type: str = "linear_4310",
    rate_hz: float = 10.0,
    save_home: str = None,
):
    """
    Stream joint positions from one or both YAM robot arms.
    
    Args:
        use_right: Whether to connect to right arm
        use_left: Whether to connect to left arm
        gripper_type: Type of gripper attached
        rate_hz: Update rate in Hz
        save_home: Optional path to save positions as numpy file on exit
    """
    print(f"\n{'='*70}")
    print(f"YAM Robot Joint Position Streamer")
    print(f"{'='*70}")
    print(f"Arms: {'Right' if use_right else ''}{' + ' if use_right and use_left else ''}{'Left' if use_left else ''}")
    print(f"Gripper Type: {gripper_type}")
    print(f"Update Rate: {rate_hz} Hz")
    print(f"{'='*70}\n")
    
    # Get gripper type enum
    gripper_type_enum = GripperType.from_string_name(gripper_type)
    
    print("Connecting to robot(s) (gravity compensation will be enabled)...")
    print("NOTE: Robot(s) will hold current position.\n")
    
    robots = {}
    
    try:
        # Connect to requested arms
        if use_right:
            print("  Connecting to RIGHT arm (can_right)...", end=" ", flush=True)
            robots["right"] = get_yam_robot(
                channel="can_right",
                gripper_type=gripper_type_enum,
                zero_gravity_mode=False,
            )
            print("OK")
        
        if use_left:
            print("  Connecting to LEFT arm (can_left)...", end=" ", flush=True)
            robots["left"] = get_yam_robot(
                channel="can_left",
                gripper_type=gripper_type_enum,
                zero_gravity_mode=False,
            )
            print("OK")
        
        print("\nStarting stream... Press Ctrl+C to stop.\n")
        time.sleep(0.5)
        
        period = 1.0 / rate_hz
        last_positions = {}
        
        while True:
            loop_start = time.time()
            
            # Read positions from all connected arms
            positions = {}
            for arm_name, robot in robots.items():
                positions[arm_name] = robot.get_joint_pos()
            
            # Clear screen and display header
            clear_screen()
            print(f"{'='*70}")
            print(f" YAM JOINT POSITIONS (streaming at {rate_hz} Hz) | Ctrl+C to exit")
            print(f"{'='*70}")
            print(f" Time: {time.strftime('%H:%M:%S')}")
            print(f"{'='*70}\n")
            
            # Display side-by-side if both arms connected
            if use_right and use_left:
                # Header
                print(f"{'RIGHT ARM':<35} | {'LEFT ARM':<35}")
                print(f"{'-'*35}-+-{'-'*35}")
                
                right_pos = positions.get("right", np.zeros(7))
                left_pos = positions.get("left", np.zeros(7))
                
                for i, name in enumerate(JOINT_NAMES):
                    right_str = format_joint_line(name, right_pos[i], i)
                    left_str = format_joint_line(name, left_pos[i], i)
                    print(f"{right_str:<35} | {left_str:<35}")
                
                print(f"\n{'-'*70}")
                print(f"{'NUMPY FORMAT'}")
                print(f"{'-'*70}")
                print(f"RIGHT: np.array({np.array2string(right_pos, precision=4, separator=', ')})")
                print(f"LEFT:  np.array({np.array2string(left_pos, precision=4, separator=', ')})")
            else:
                # Single arm display
                arm_name = "right" if use_right else "left"
                arm_label = "RIGHT" if use_right else "LEFT"
                pos = positions[arm_name]
                
                print(f"{arm_label} ARM")
                print(f"{'-'*40}")
                
                for i, name in enumerate(JOINT_NAMES):
                    print(format_joint_line(name, pos[i], i))
                
                print(f"\n{'-'*40}")
                print(f"NUMPY: np.array({np.array2string(pos, precision=4, separator=', ')})")
            
            print(f"\n{'='*70}")
            print(f" '!' indicates joint near limit")
            print(f"{'='*70}")
            
            # Store latest positions for saving
            last_positions = positions
            
            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\nStopping stream...")
        
        # Save positions if requested
        if save_home and last_positions:
            save_dict = {f"{k}_arm": v for k, v in last_positions.items()}
            np.savez(save_home, **save_dict)
            print(f"Saved positions to: {save_home}")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPossible issues:")
        print("  - Is the CAN interface up? Run: ip link show | grep can")
        print("  - Is the robot powered on?")
        print("  - Is another process using the CAN interface?")
        raise
    finally:
        # Close all robot connections
        for arm_name, robot in robots.items():
            try:
                robot.close()
                print(f"Closed {arm_name} arm connection.")
            except:
                pass
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Stream YAM robot joint positions from both arms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream both arms (default)
  python read_yam_joints.py
  
  # Stream right arm only
  python read_yam_joints.py --right-only
  
  # Stream left arm only  
  python read_yam_joints.py --left-only
  
  # Stream at 20Hz
  python read_yam_joints.py --rate 20
  
  # Save positions on exit
  python read_yam_joints.py --save-home positions.npz
  
  # Use specific gripper type
  python read_yam_joints.py --gripper linear_4310
        """
    )
    
    parser.add_argument(
        "--right-only",
        action="store_true",
        help="Only connect to right arm"
    )
    parser.add_argument(
        "--left-only",
        action="store_true",
        help="Only connect to left arm"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Update rate in Hz (default: 10)"
    )
    parser.add_argument(
        "--gripper",
        type=str,
        default="linear_4310",
        choices=["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"],
        help="Gripper type (default: linear_4310)"
    )
    parser.add_argument(
        "--save-home",
        type=str,
        default=None,
        help="Save final joint positions to numpy file (.npz) on exit"
    )
    
    args = parser.parse_args()
    
    # Determine which arms to use
    if args.right_only and args.left_only:
        parser.error("Cannot specify both --right-only and --left-only")
    
    use_right = not args.left_only
    use_left = not args.right_only
    
    stream_joint_positions(
        use_right=use_right,
        use_left=use_left,
        gripper_type=args.gripper,
        rate_hz=args.rate,
        save_home=args.save_home,
    )


if __name__ == "__main__":
    main()
