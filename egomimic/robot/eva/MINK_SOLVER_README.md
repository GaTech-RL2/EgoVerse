# Mink Kinematics Solver for Eva Robot

This directory contains a mink-based inverse kinematics solver for the Eva (ARX X5) robot arm.

## Overview

We've implemented two new solver classes:

1. **`MinkKinematicsSolver`** (in `egomimic/robot/kinematics.py`)
   - Generic kinematics solver using mink (MuJoCo-based optimization IK)
   - Provides similar interface to the existing `KinematicsSolver` (TracIK)
   - Works with any robot that has a MuJoCo XML model

2. **`EvaMinkKinematicsSolver`** (in `egomimic/robot/eva/eva_kinematics.py`)
   - Eva-specific implementation extending `MinkKinematicsSolver`
   - Pre-configured for ARX X5 robot (6 DOF arm)
   - Includes helper methods like `ik_with_retries()`

## Key Features

### MinkKinematicsSolver
- **Optimization-based IK**: Uses quadratic programming to solve IK
- **Task specification**: Supports position and orientation tasks
- **Limits**: Enforces joint position and velocity limits
- **Configurable**: Adjustable convergence tolerances, solver choice, etc.

### EvaMinkKinematicsSolver
- **Eva-specific configuration**: Pre-configured joint names and limits for ARX X5
- **Multiple retries**: `ik_with_retries()` method for difficult configurations
- **Base transform support**: Can handle mobile base transformations

## Installation

Make sure you have mink installed:

```bash
pip install mink
```

This will also install MuJoCo and other dependencies.

## Usage

### Basic Usage

```python
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
import numpy as np
from scipy.spatial.transform import Rotation as R

# Initialize solver (requires MuJoCo XML file)
solver = EvaMinkKinematicsSolver(
    urdf_path="path/to/x5_scene_mod.xml",
    eef_link_name="gripper",
    eef_frame_type="site"
)

# Current joint configuration
current_joints = np.array([0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

# Target pose
target_pos = np.array([0.5, 0.0, 0.8])  # xyz in meters
target_rot = R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()

# Solve IK
solution = solver.ik(target_pos, target_rot, current_joints)

if solution is not None:
    print(f"IK solution: {solution}")
else:
    print("IK failed to converge")
```

### Forward Kinematics

```python
# Compute FK
pos, rot = solver.fk(current_joints)
print(f"End-effector position: {pos}")
print(f"End-effector orientation: {rot.as_euler('xyz', degrees=True)}")
```

### IK with Retries

For difficult configurations, use the retry mechanism:

```python
solution = solver.ik_with_retries(
    target_pos,
    target_rot, 
    current_joints,
    num_retries=5
)
```

## API Reference

### MinkKinematicsSolver

#### Constructor Parameters
- `urdf_path` (str): Path to MuJoCo XML file
- `base_link_name` (str): Name of base link
- `eef_link_name` (str): Name of end-effector link/site
- `num_joints` (int): Number of joints to control
- `joint_names` (List[str]): Ordered list of joint names
- `eef_frame_type` (str): "site" or "body"
- `velocity_limits` (dict): Joint velocity limits in rad/s
- `solver` (str): QP solver ("daqp", "quadprog", "proxqp", etc.)
- `max_iterations` (int): Maximum IK iterations (default: 100)
- `position_tolerance` (float): Position convergence tolerance in m (default: 1e-3)
- `orientation_tolerance` (float): Orientation convergence tolerance in rad (default: 1e-3)

#### Methods
- `ik(pos_xyz, rot_mat, cur_jnts, dt=0.01)`: Solve inverse kinematics
  - Returns: Joint solution array or None if failed
  
- `fk(jnts)`: Solve forward kinematics
  - Returns: (pos, rot) tuple where rot is scipy Rotation object

### EvaMinkKinematicsSolver

#### Constructor Parameters
- `urdf_path` (str): Path to MuJoCo XML file
- `eef_link_name` (str): End-effector frame name (default: "gripper")
- `eef_frame_type` (str): Frame type (default: "site")
- `velocity_limits` (dict): Optional velocity limits
- `solver` (str): QP solver (default: "daqp")
- `max_iterations` (int): Max iterations (default: 100)
- `position_tolerance` (float): Position tolerance (default: 1e-3)
- `orientation_tolerance` (float): Orientation tolerance (default: 1e-3)

#### Additional Methods
- `ik_with_retries(pos_xyz, rot_mat, cur_jnts, num_retries=3, dt=0.01)`: IK with multiple attempts
- `set_base_transform(transform)`: Set base transformation matrix

## MuJoCo XML Requirements

The solver requires a MuJoCo XML scene file. The XML should:

1. Include the robot URDF or MJCF model
2. Define a site or body for the end-effector with name matching `eef_link_name`
3. Have joints named according to `joint_names`
4. Include proper mesh paths

Example minimal scene:

```xml
<mujoco model="x5_scene">
  <compiler angle="radian" meshdir="meshes"/>
  
  <worldbody>
    <body name="base_link" pos="0 0 0.5">
      <!-- Robot structure here -->
      <site name="gripper" pos="..." quat="..."/>
    </body>
  </worldbody>
  
  <actuator>
    <position name="joint1" joint="joint1" kp="50"/>
    <!-- More actuators -->
  </actuator>
</mujoco>
```

## Testing

Run the test script:

```bash
cd egomimic/robot/eva
python test_mink_solver.py
```

## Comparison: TracIK vs Mink

| Feature | TracIK | Mink |
|---------|--------|------|
| Speed | Fast (analytical + numerical) | Slower (optimization-based) |
| Accuracy | Good | Excellent (configurable) |
| Constraints | Basic limits | Position, velocity, collision avoidance |
| Setup | URDF only | Requires MuJoCo XML |
| Dependencies | trac_ik | mink + MuJoCo |

## When to Use Mink

Use `EvaMinkKinematicsSolver` when you need:
- **High accuracy**: Tight convergence tolerances
- **Complex constraints**: Collision avoidance, task priorities
- **Smooth trajectories**: Velocity limiting for smooth motion
- **MuJoCo integration**: Already using MuJoCo for simulation

Use `EvaKinematicsSolver` (TracIK) when you need:
- **Speed**: Real-time performance critical
- **Simple setup**: Just have URDF file
- **Standard IK**: No special constraints needed

## Troubleshooting

### "mink not found"
Install mink: `pip install mink`

### "Cannot load XML file"
Make sure you're providing a MuJoCo XML file, not a URDF. You may need to convert your URDF to MJCF format.

### IK not converging
- Increase `max_iterations`
- Relax tolerances (`position_tolerance`, `orientation_tolerance`)
- Use `ik_with_retries()` for difficult configurations
- Check if target is reachable

### Wrong end-effector frame
- Verify `eef_link_name` matches a site or body in your XML
- Check `eef_frame_type` is correct ("site" or "body")

## References

- **mink**: https://github.com/kevinzakka/mink
- **MuJoCo**: https://mujoco.org/
- **Paper**: "A micro lie theory for state estimation in robotics" (for SE3 transforms)

## License

Same as parent project.

