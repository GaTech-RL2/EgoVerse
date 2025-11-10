import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm, trange

from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

NORMAL_VIS_LEN = 0.05  # meters
ARIA_SLAM_TO_CENTER_LEN = 0.08 # meters

Z_FRONT_TO_Z_UP = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]
])

vrs_sample_path = "./data/roger_walking_test.vrs"
annotation_dir_name = "mps_" + os.path.basename(vrs_sample_path).replace(".vrs", "_vrs")
mps_sample_path = os.path.join(os.path.dirname(vrs_sample_path), annotation_dir_name)

# Load the VRS file
vrsfile = os.path.join(mps_sample_path, "sample.vrs")

# Trajectory, global points, and online calibration
closed_loop_trajectory = os.path.join(
    mps_sample_path, "slam", "closed_loop_trajectory.csv"
)
global_points = os.path.join(mps_sample_path, "slam", "semidense_points.csv.gz")
online_calibrations_path = os.path.join(mps_sample_path, "slam", "online_calibration.jsonl")

# Eye gaze
generalized_eye_gaze_path = os.path.join(
    mps_sample_path, "eye_gaze", "general_eye_gaze.csv"
)

# Hand tracking
hand_tracking_results_path = os.path.join(
    mps_sample_path, "hand_tracking", "hand_tracking_results.csv"
)

# Create data provider and get T_device_rgb
provider = data_provider.create_vrs_data_provider(vrs_sample_path)
# Since we want to display the position of the RGB camera, we are querying its relative location
# from the device and will apply it to the device trajectory.

## Load trajectory and global points
mps_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory)
points = mps.read_global_point_cloud(global_points)

mps_trajectory_ns_timestamp_arr = np.zeros(len(mps_trajectory), dtype=np.uint64)

for i in range(len(mps_trajectory)):
    mps_trajectory_ns_timestamp_arr[i] = mps_trajectory[i].tracking_timestamp.total_seconds() * 1e9

## Load online calibration file
online_calibrations = mps.read_online_calibration(online_calibrations_path)

## Load eyegaze
generalized_eye_gazes = mps.read_eyegaze(generalized_eye_gaze_path)

## Load hand tracking
hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
    hand_tracking_results_path
)

# Loaded data must be not empty
# TODO(roger): perosnalized eye gaze is not loaded
assert(
    len(mps_trajectory) != 0 and
    len(points) != 0 and
    len(online_calibrations) !=0 and
    len(generalized_eye_gazes) != 0 and
    len(hand_tracking_results) != 0)


# Or you can load the whole mps output with MpsDataProvider
mps_data_provider = mps.MpsDataProvider(mps.MpsDataPathsProvider(mps_sample_path).get_data_paths())

assert(mps_data_provider.has_general_eyegaze() and
    mps_data_provider.has_open_loop_poses() and
    mps_data_provider.has_closed_loop_poses() and
    mps_data_provider.has_online_calibrations() and
    mps_data_provider.has_semidense_point_cloud() and
    mps_data_provider.has_hand_tracking_results())

# Get the MPS service versions
# print(f"slam_version: {mps_data_provider.get_slam_version()}")
# print(f"eyegaze_version: {mps_data_provider.get_eyegaze_version()}")
# print(f"hand_tracking_version: {mps_data_provider.get_hand_tracking_version()}")

device_calibration = provider.get_device_calibration()

time_domain: TimeDomain = TimeDomain.DEVICE_TIME
time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST

# Get stream ids, stream labels, stream timestamps, and camera calibrations for RGB and SLAM cameras
stream_ids: Dict[str, StreamId] = {
    "rgb": StreamId("214-1"),
    "slam-left": StreamId("1201-1"),
    "slam-right": StreamId("1201-2"),
}
stream_labels: Dict[str, str] = {
    key: provider.get_label_from_stream_id(stream_id)
    for key, stream_id in stream_ids.items()
}
stream_timestamps_ns: Dict[str, List[int]] = {
    key: provider.get_timestamps_ns(stream_id, time_domain)
    for key, stream_id in stream_ids.items()
}
camera_calibrations = {
    key: device_calibration.get_camera_calib(stream_label)
    for key, stream_label in stream_labels.items()
}
for key, camera_calibration in camera_calibrations.items():
    assert camera_calibration is not None, f"no camera calibration for {key}"

# Get device calibration and transform from device to sensor
device_calibration = provider.get_device_calibration()

def undistort_to_linear(provider, stream_ids, raw_image, camera_label="rgb"):
    camera_label = provider.get_label_from_stream_id(stream_ids[camera_label])
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    warped = calibration.get_linear_camera_calibration(
        480, 640, 133.25430222 * 2, camera_label, calib.get_transform_device_camera()
    )
    warped_image = calibration.distort_by_calibration(raw_image, warped, calib)
    warped_rot = np.rot90(warped_image, k=3)
    return warped_rot

# ================================================
# Begin main function
# ================================================

import plotly.graph_objects as go
import numpy as np

def visualize_hand_landmarks_3d(num_frames=50):
    """
    Create a 3D visualization of hand landmarks and head pose with animation across multiple frames.
    
    Args:
        num_frames: Number of frames to include in the animation
    """
    # Prepare data for animation
    frames = []
    
    # Limit number of frames to available data
    max_frames = min(num_frames, len(stream_timestamps_ns["rgb"]))
    
    # Helper function to extract coordinate axes from transformation matrix
    def get_axes(mat, scale=0.1):
        R = mat[:3, :3]
        pos = mat[:3, 3]
        x_axis = pos + R[:, 0] * scale
        y_axis = pos + R[:, 1] * scale
        z_axis = pos + R[:, 2] * scale
        return pos, x_axis, y_axis, z_axis
    
    # Initialize plotly with most populated frame
    tracing_frames_idx = None
    tracing_frames_num_feat = -1

    init_rotation_world_device = mps_trajectory[0].transform_world_device.rotation().to_matrix() @ Z_FRONT_TO_Z_UP
    init_translation_world_device = mps_trajectory[0].transform_world_device.translation()
    
    # Process each frame
    for frame_idx in range(max_frames):
        cur_frame_num_feat = 0
        # Get landmarks data for current frame
        sample_timestamp_ns = stream_timestamps_ns["rgb"][frame_idx]

        sample_frames = {
            key: provider.get_image_data_by_time_ns(
                stream_id, sample_timestamp_ns, time_domain, time_query_closest
            )[0]
            for key, stream_id in stream_ids.items()
        }

        rgb_image = sample_frames["rgb"].to_numpy_array()

        if False:
            # rgb_image = rgb_image[:, :, ::-1]
            rgb_image = undistort_to_linear(provider, stream_ids, rgb_image)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("rgb", rgb_image)
            cv2.waitKey(33)

        frame_hand_tracking_result = mps_data_provider.get_hand_tracking_result(
            sample_timestamp_ns, time_query_closest
        )
        
        # Get head pose from trajectory
        timestamp_abs_offset = np.abs(mps_trajectory_ns_timestamp_arr - sample_timestamp_ns)
        closest_idx = np.argmin(timestamp_abs_offset)
        # timestamp_abs_offset is in nanoseconds
        if timestamp_abs_offset[closest_idx] > 0.1 * 1e9:
            print("Skipping frame {}".format(frame_idx))
            continue
        
        rotation_world_device = mps_trajectory[closest_idx].transform_world_device.rotation().to_matrix() @ Z_FRONT_TO_Z_UP
        translation_world_device = mps_trajectory[closest_idx].transform_world_device.translation()

        # Translate head poses so that they are w.r.t. to the initial head pose
        rotation_world_device = init_rotation_world_device.T @ rotation_world_device
        translation_world_device = np.zeros_like(translation_world_device)
        
        # Create head transformation matrix
        head_mat = np.eye(4)
        head_mat[:3, :3] = rotation_world_device
        head_mat[:3, 3] = translation_world_device
        
        # Get head pose and axes
        head_pos, head_x, head_y, head_z = get_axes(head_mat)
        
        # Get hand landmarks
        left_landmarks = frame_hand_tracking_result.left_hand.landmark_positions_device if frame_hand_tracking_result.left_hand else []
        right_landmarks = frame_hand_tracking_result.right_hand.landmark_positions_device if frame_hand_tracking_result.right_hand else []
        
        # Convert to numpy arrays for easier handling
        left_array = np.array([landmark for landmark in left_landmarks]) if left_landmarks else np.empty((0, 3))
        right_array = np.array([landmark for landmark in right_landmarks]) if right_landmarks else np.empty((0, 3))

        # Get 4x4 poses
        left_hand_wrist_pose = frame_hand_tracking_result.left_hand.transform_device_wrist.to_matrix()
        right_hand_wrist_pose = frame_hand_tracking_result.right_hand.transform_device_wrist.to_matrix()

        # Convert to x-front, y-right, z-up coordinate system
        LEFT_AVALIABLE = len(left_array) > 0
        RIGHT_AVAILABLE = len(right_array) > 0

        # Apply same transformations to left_hand_wrist_pose
        # Create coordinate transformation matrix: [z, y, -x] mapping
        coord_transform = np.eye(4)
        coord_transform[:3, :3] = np.array([
            [0, 0, 1],   # new x = old z
            [0, 1, 0],   # new y = old y  
            [-1, 0, 0]   # new z = -old x
        ])
        
        # Create translation matrix for y-axis offset
        translation_transform = np.eye(4)
        translation_transform[1, 3] = ARIA_SLAM_TO_CENTER_LEN
        
        # Create rotation matrix
        rotation_transform = np.eye(4)
        rotation_transform[:3, :3] = rotation_world_device.T

        if LEFT_AVALIABLE:
            left_array = left_array[:, [2, 1, 0]] * [1, 1, -1]
            # Before process: coordinate is relative to camera-rgb on the left side of the glass
            left_array[:, 1] += ARIA_SLAM_TO_CENTER_LEN
            # Apply rotation to align with world coordinate system
            left_array = left_array @ rotation_world_device.T
            
            # Apply transformations to pose matrix in correct order
            left_hand_wrist_pose = rotation_transform @ translation_transform @ coord_transform @ left_hand_wrist_pose
        
        if RIGHT_AVAILABLE:
            right_array = right_array[:, [2, 1, 0]] * [1, 1, -1]
            # Before process: coordinate is relative to camera-rgb on the right side of the glass
            right_array[:, 1] += ARIA_SLAM_TO_CENTER_LEN
            # Apply rotation to align with world coordinate system
            right_array = right_array @ rotation_world_device.T

            # Apply same transformations to right_hand_wrist_pose
            right_hand_wrist_pose = rotation_transform @ translation_transform @ coord_transform @ right_hand_wrist_pose
        
        # Create traces for this frame
        frame_traces = []
        
        # Add head position marker
        frame_traces.append(go.Scatter3d(
            x=[head_pos[0]],
            y=[head_pos[1]],
            z=[head_pos[2]],
            mode='markers',
            name='Head Position',
            marker=dict(size=8, color='blue', opacity=0.8)
        ))
        
        # Add head coordinate axes
        axis_colors = ['red', 'green', 'blue']  # x, y, z axes colors
        axis_names = ['X', 'Y', 'Z']
        head_axes = [head_x, head_y, head_z]
        
        for i, (axis_end, color, axis_name) in enumerate(zip(head_axes, axis_colors, axis_names)):
            frame_traces.append(go.Scatter3d(
                x=[head_pos[0], axis_end[0]],
                y=[head_pos[1], axis_end[1]],
                z=[head_pos[2], axis_end[2]],
                mode='lines',
                name=f'Head {axis_name} Axis',
                line=dict(color=color, width=3, dash='solid')
            ))
        
        # Add left hand landmarks
        if len(left_array) > 0:
            cur_frame_num_feat += 1
            frame_traces.append(go.Scatter3d(
                x=left_array[:, 0],
                y=left_array[:, 1],
                z=left_array[:, 2],
                mode='markers',
                name='Left Hand',
                marker=dict(size=6, color='green', opacity=0.8)
            ))
        
        # Add right hand landmarks
        if len(right_array) > 0:
            cur_frame_num_feat += 1
            frame_traces.append(go.Scatter3d(
                x=right_array[:, 0],
                y=right_array[:, 1],
                z=right_array[:, 2],
                mode='markers',
                name='Right Hand',
                marker=dict(size=6, color='red', opacity=0.8)
            ))
        
        # Add the frame to our frames list
        frames.append(go.Frame(data=frame_traces, name=str(frame_idx)))

        if cur_frame_num_feat > tracing_frames_num_feat:
            tracing_frames_idx = frame_idx
            tracing_frames_num_feat = cur_frame_num_feat
    
    # Create the figure and add the initial (first) frame data
    fig = go.Figure()
    
    # Add initial frame data to the figure
    if frames and frames[tracing_frames_idx].data:
        for trace in frames[tracing_frames_idx].data:
            fig.add_trace(trace)
    
    # Set frames for animation
    fig.frames = frames
    
    # Configure the layout for proper 3D visualization with animation controls
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=800,
        title="Hand Landmarks and Head Pose 3D Animation",
        showlegend=True,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [str(k)],
                        {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}
                    ],
                    'label': str(k),
                    'method': 'animate'
                } for k in range(max_frames)
            ]
        }]
    )
    
    return fig

# Call the function to create and display the visualization with 50 frames
fig = visualize_hand_landmarks_3d(1000)
fig.show()

