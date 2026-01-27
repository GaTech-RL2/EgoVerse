import sys
import numpy as np
sys.path.insert(0, '/home/robot/i2rt')  # Adjust path if needed
sys.path.insert(0, '/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/eva')

from scipy.spatial.transform import Rotation as R

# ============================================================================
# HDF5 VALUES FROM corn_bowl_2/demo_0.hdf5 t=0
# ============================================================================
HDF5_LEFT_XYZ = np.array([0.25190294, -0.00263565, 0.14809819])
HDF5_LEFT_YPR = np.array([3.1414516, 1.4613258, 3.1169276])
HDF5_RIGHT_XYZ = np.array([0.2441737, -0.00081471, 0.149414])
HDF5_RIGHT_YPR = np.array([3.1168997, 1.505357, 3.1173124])

# Live robot home pose from inference logs
LIVE_LEFT_XYZ = np.array([0.24438366, -0.00329742, 0.17996436])
LIVE_RIGHT_XYZ = np.array([0.24363534, 0.00104101, 0.18250771])

# ============================================================================
# SETUP
# ============================================================================
print("="*70)
print("DIAGNOSTIC: EE Pose Mismatch Investigation")
print("="*70)

try:
    from robot_interface import YAMInterface
    
    print("\nInitializing YAMInterface with both arms...")
    
    # Initialize both arms at once using YAMInterface
    robot = YAMInterface(
        arms=["left", "right"],
        gripper_type="linear_4310",
        interfaces={"left": "can_left", "right": "can_right"},
        cameras_cfg={},  # No cameras needed for this test
        zero_gravity_mode=False,  # Control mode - robot will actively move
    )
    
    print(f"Gripper type: linear_4310")
    
    # ============================================================================
    # TEST 1: FK of HOME_POSITION using YAMInterface kinematics
    # ============================================================================
    print("\n" + "="*70)
    print("TEST 1: FK of HOME_POSITION")
    print("="*70)
    
    HOME_JOINTS = np.array([0.0, 0.05, 0.05, 0.0, 0.0, 0.0])
    print(f"HOME_POSITION joints: {HOME_JOINTS}")
    
    # Use YAMInterface's internal kinematics (same for both arms)
    T_home = robot.kinematics["left"].fk(HOME_JOINTS)
    home_xyz = T_home[:3, 3]
    home_ypr = R.from_matrix(T_home[:3, :3]).as_euler("ZYX", degrees=False)
    
    print(f"\nFK(HOME_POSITION) via YAMInterface:")
    print(f"  XYZ: {home_xyz}")
    print(f"  YPR: {home_ypr}")
    print(f"  Z (mm): {home_xyz[2] * 1000:.2f}")
    
    print(f"\nComparison to HDF5 t=0:")
    print(f"  HDF5 Left Z:  {HDF5_LEFT_XYZ[2] * 1000:.2f} mm")
    print(f"  HDF5 Right Z: {HDF5_RIGHT_XYZ[2] * 1000:.2f} mm")
    print(f"  HOME FK Z:    {home_xyz[2] * 1000:.2f} mm")
    print(f"  Diff (HOME - HDF5 Left): {(home_xyz[2] - HDF5_LEFT_XYZ[2]) * 1000:.2f} mm")
    
    print(f"\nComparison to Live robot (from inference logs):")
    print(f"  Live Left Z:  {LIVE_LEFT_XYZ[2] * 1000:.2f} mm")
    print(f"  Live Right Z: {LIVE_RIGHT_XYZ[2] * 1000:.2f} mm")
    print(f"  Diff (HOME - Live Left): {(home_xyz[2] - LIVE_LEFT_XYZ[2]) * 1000:.2f} mm")
    
    # ============================================================================
    # TEST 2: Read actual robot state using YAMInterface methods
    # ============================================================================
    print("\n" + "="*70)
    print("TEST 2: Read actual robot joint positions and compute FK")
    print("="*70)
    
    # Move both arms to home position
    print("\nMoving both arms to home position...")
    robot.set_home()
    print("Home position reached.")
    
    # Read positions from each arm using YAMInterface methods
    for arm_name in ["left", "right"]:
        try:
            # Get joint positions
            joints = robot.get_joints(arm_name)
            print(f"\n{arm_name.upper()} ARM:")
            print(f"  Joint positions: {joints}")
            
            # Use YAMInterface's get_pose() method for FK
            xyz, rot = robot.get_pose(arm_name, se3=False)
            ypr = rot.as_euler("ZYX", degrees=False)
            
            # Alternative: use get_pose_6d() which returns [x,y,z,yaw,pitch,roll]
            pose_6d = robot.get_pose_6d(arm_name)
            
            print(f"  FK result (via robot.get_pose()):")
            print(f"    XYZ: {xyz}")
            print(f"    YPR: {ypr}")
            print(f"    Z (mm): {xyz[2] * 1000:.2f}")
            print(f"  FK result (via robot.get_pose_6d()):")
            print(f"    pose_6d: {pose_6d}")
            
            if arm_name == "left":
                print(f"  vs HDF5 Left XYZ diff (mm): {(xyz - HDF5_LEFT_XYZ) * 1000}")
                print(f"  vs Live Left XYZ diff (mm): {(xyz - LIVE_LEFT_XYZ) * 1000}")
            else:
                print(f"  vs HDF5 Right XYZ diff (mm): {(xyz - HDF5_RIGHT_XYZ) * 1000}")
                print(f"  vs Live Right XYZ diff (mm): {(xyz - LIVE_RIGHT_XYZ) * 1000}")
                
        except Exception as e:
            print(f"\n{arm_name.upper()} ARM: Error reading - {e}")
    
    # ============================================================================
    # TEST 3: Inverse check - what joints produce HDF5 positions?
    # ============================================================================
    print("\n" + "="*70)
    print("TEST 3: What EE pose does HDF5 t=0 correspond to?")
    print("="*70)
    
    print(f"\nHDF5 t=0 Left arm:")
    print(f"  XYZ: {HDF5_LEFT_XYZ} (Z = {HDF5_LEFT_XYZ[2]*1000:.2f} mm)")
    print(f"  YPR: {HDF5_LEFT_YPR}")
    
    print(f"\nHDF5 t=0 Right arm:")
    print(f"  XYZ: {HDF5_RIGHT_XYZ} (Z = {HDF5_RIGHT_XYZ[2]*1000:.2f} mm)")
    print(f"  YPR: {HDF5_RIGHT_YPR}")
    
    print(f"\nLive robot home (from inference logs):")
    print(f"  Left XYZ:  {LIVE_LEFT_XYZ} (Z = {LIVE_LEFT_XYZ[2]*1000:.2f} mm)")
    print(f"  Right XYZ: {LIVE_RIGHT_XYZ} (Z = {LIVE_RIGHT_XYZ[2]*1000:.2f} mm)")
    
    print(f"\n*** Z POSITION DISCREPANCY ***")
    print(f"  Left arm:  Live Z ({LIVE_LEFT_XYZ[2]*1000:.2f}mm) - HDF5 Z ({HDF5_LEFT_XYZ[2]*1000:.2f}mm) = {(LIVE_LEFT_XYZ[2]-HDF5_LEFT_XYZ[2])*1000:.2f} mm")
    print(f"  Right arm: Live Z ({LIVE_RIGHT_XYZ[2]*1000:.2f}mm) - HDF5 Z ({HDF5_RIGHT_XYZ[2]*1000:.2f}mm) = {(LIVE_RIGHT_XYZ[2]-HDF5_RIGHT_XYZ[2])*1000:.2f} mm")
    
    # ============================================================================
    # TEST 4: Verify YAMInterface kinematics consistency
    # ============================================================================
    print("\n" + "="*70)
    print("TEST 4: Verify YAMInterface kinematics consistency between arms")
    print("="*70)
    
    # Both arms should use the same kinematics model
    T_left = robot.kinematics["left"].fk(HOME_JOINTS)
    T_right = robot.kinematics["right"].fk(HOME_JOINTS)
    
    left_z = T_left[2, 3]
    right_z = T_right[2, 3]
    
    print(f"  Left arm kinematics FK(HOME) Z:  {left_z*1000:.2f} mm")
    print(f"  Right arm kinematics FK(HOME) Z: {right_z*1000:.2f} mm")
    print(f"  Difference: {abs(left_z - right_z)*1000:.4f} mm")
    
    if abs(left_z - right_z) < 0.0001:
        print("  ✓ Both arms use identical kinematics model")
    else:
        print("  ✗ WARNING: Arms have different kinematics models!")
    
    # Clean up
    robot.close()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
If FK(HOME_POSITION) Z matches Live robot Z but NOT HDF5 Z:
  -> The HDF5 demos were NOT recorded starting from this HOME_POSITION
  -> Or a different FK site/model was used during recording

If FK(HOME_POSITION) Z matches HDF5 Z but NOT Live robot Z:
  -> The robot is not actually at HOME_POSITION
  -> Check if set_home() is being called correctly

If FK(HOME_POSITION) Z matches neither:
  -> Possible model/site_name mismatch between recording and inference
""")

except ImportError as e:
    print(f"Could not import robot modules: {e}")
    print("Skipping robot tests.")
except Exception as e:
    import traceback
    print(f"Error during robot test: {e}")
    traceback.print_exc()
    print("Skipping robot tests.")
