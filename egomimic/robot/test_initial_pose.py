import sys
import numpy as np
sys.path.insert(0, '/home/robot/i2rt')  # Adjust path if needed
sys.path.insert(0, '/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/eva')

from scipy.spatial.transform import Rotation as R
from i2rt.robots.kinematics import Kinematics as I2RTKinematics
from i2rt.robots.utils import GripperType

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

gripper_type = GripperType.from_string_name("linear_4310")
model_path = gripper_type.get_xml_path()
print(f"\nGripper type: linear_4310")
print(f"Model path: {model_path}")

kin = I2RTKinematics(xml_path=model_path, site_name="grasp_site")

# ============================================================================
# TEST 1: FK of HOME_POSITION
# ============================================================================
print("\n" + "="*70)
print("TEST 1: FK of HOME_POSITION")
print("="*70)

HOME_JOINTS = np.array([0.0, 0.05, 0.05, 0.0, 0.0, 0.0])
print(f"HOME_POSITION joints: {HOME_JOINTS}")

T_home = kin.fk(HOME_JOINTS)
home_xyz = T_home[:3, 3]
home_ypr = R.from_matrix(T_home[:3, :3]).as_euler("ZYX", degrees=False)

print(f"\nFK(HOME_POSITION):")
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
# TEST 2: Read actual robot state
# ============================================================================
print("\n" + "="*70)
print("TEST 2: Read actual robot joint positions and compute FK")
print("="*70)

try:
    from i2rt.robots.get_robot import get_yam_robot
    
    print("Connecting to robot arms...")
    
    # Try to read from both arms
    for arm_name, channel in [("left", "can0"), ("right", "can1")]:
        try:
            robot = get_yam_robot(
                channel=channel,
                gripper_type=gripper_type,
                zero_gravity_mode=True,  # Read-only, don't move
            )
            
            joints = robot.get_joint_pos()
            print(f"\n{arm_name.upper()} ARM:")
            print(f"  Joint positions: {joints}")
            
            T = kin.fk(joints[:6])
            xyz = T[:3, 3]
            ypr = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=False)
            
            print(f"  FK result:")
            print(f"    XYZ: {xyz}")
            print(f"    YPR: {ypr}")
            print(f"    Z (mm): {xyz[2] * 1000:.2f}")
            
            if arm_name == "left":
                print(f"  vs HDF5 Left XYZ diff (mm): {(xyz - HDF5_LEFT_XYZ) * 1000}")
                print(f"  vs Live Left XYZ diff (mm): {(xyz - LIVE_LEFT_XYZ) * 1000}")
            else:
                print(f"  vs HDF5 Right XYZ diff (mm): {(xyz - HDF5_RIGHT_XYZ) * 1000}")
                print(f"  vs Live Right XYZ diff (mm): {(xyz - LIVE_RIGHT_XYZ) * 1000}")
                
        except Exception as e:
            print(f"\n{arm_name.upper()} ARM: Could not connect - {e}")

except ImportError as e:
    print(f"Could not import robot modules: {e}")
    print("Skipping live robot test.")

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
# TEST 4: Check different site names
# ============================================================================
print("\n" + "="*70)
print("TEST 4: Check if site_name affects FK output")
print("="*70)

for site in ["grasp_site", "attachment_site", "end_effector"]:
    try:
        kin_test = I2RTKinematics(xml_path=model_path, site_name=site)
        T = kin_test.fk(HOME_JOINTS)
        xyz = T[:3, 3]
        print(f"  site_name='{site}': Z = {xyz[2]*1000:.2f} mm")
    except Exception as e:
        print(f"  site_name='{site}': ERROR - {e}")

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