import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_uid = p.loadURDF(
    "assets/franka_panda/franka_panda_arm_marker.urdf", cubeStartPos, cubeStartOrientation
)
p.stepSimulation()

num_joints = p.getNumJoints(robot_uid)
print(num_joints)

neutral_positions = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]

# neutral_positions = [0.09162008114028396, -0.19826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 1.30396583422025, 0.8480939705504309]

for i in range(7):
    p.resetJointState(bodyUniqueId=robot_uid, jointIndex=i, targetValue=neutral_positions[i])

for i in range(9, 11):
    p.resetJointState(bodyUniqueId=robot_uid, jointIndex=i, targetValue=0.04)

marker_joint = num_joints - 1
marker_link = marker_joint

print(p.getJointInfo(bodyUniqueId=robot_uid, jointIndex=marker_joint))
print(p.getLinkState(bodyUniqueId=robot_uid, linkIndex=marker_link))

# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
p.resetDebugVisualizerCamera(
    cameraDistance=1, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 1.0]
)
input()
# p.disconnect()
