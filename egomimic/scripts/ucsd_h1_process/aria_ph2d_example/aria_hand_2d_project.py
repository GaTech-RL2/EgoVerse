import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm, trange
import argparse

from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

NORMAL_VIS_LEN = 0.05  # meters

def main(vrs_sample_path: str):
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

    # ================================================
    # Begin helper visualization functions
    # ================================================
    def get_T_device_sensor(device_calibration: calibration.DeviceCalibration, key: str):
        return device_calibration.get_transform_device_sensor(stream_labels[key])

    # Helper functions for reprojection and plotting
    def get_point_reprojection(
        point_position_device: np.array, device_calibration: calibration.DeviceCalibration, key: str
    ) -> Optional[np.array]:
        point_position_camera = get_T_device_sensor(device_calibration, key).inverse() @ point_position_device
        point_position_pixel = camera_calibrations[key].project(point_position_camera)
        return point_position_pixel

    def get_landmark_pixels(key: str, hand_tracking_result: mps.hand_tracking.HandTrackingResult, device_calibration: calibration.DeviceCalibration) -> np.array:
        left_wrist = None
        left_palm = None
        left_landmarks = None
        right_wrist = None
        right_palm = None
        right_landmarks = None
        left_wrist_normal_tip = None
        left_palm_normal_tip = None
        right_wrist_normal_tip = None
        right_palm_normal_tip = None
        if hand_tracking_result.left_hand:
            left_landmarks = [
                get_point_reprojection(landmark, device_calibration, key)
                for landmark in hand_tracking_result.left_hand.landmark_positions_device
            ]
            left_wrist = get_point_reprojection(
                hand_tracking_result.left_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.WRIST)
                ],
                device_calibration,
                key,
            )
            left_palm = get_point_reprojection(
                hand_tracking_result.left_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                ],
                device_calibration,
                key,
            )
            if hand_tracking_result.left_hand.wrist_and_palm_normal_device is not None:
                left_wrist_normal_tip = get_point_reprojection(
                    hand_tracking_result.left_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.WRIST)
                    ]
                    + hand_tracking_result.left_hand.wrist_and_palm_normal_device.wrist_normal_device
                    * NORMAL_VIS_LEN,
                    device_calibration,
                    key,
                )
                left_palm_normal_tip = get_point_reprojection(
                    hand_tracking_result.left_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                    ]
                    + hand_tracking_result.left_hand.wrist_and_palm_normal_device.palm_normal_device
                    * NORMAL_VIS_LEN,
                    device_calibration,
                    key,
                )
        if hand_tracking_result.right_hand:
            right_landmarks = [
                get_point_reprojection(landmark, device_calibration, key)
                for landmark in hand_tracking_result.right_hand.landmark_positions_device
            ]
            right_wrist = get_point_reprojection(
                hand_tracking_result.right_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.WRIST)
                ],
                device_calibration,
                key,
            )
            right_palm = get_point_reprojection(
                hand_tracking_result.right_hand.landmark_positions_device[
                    int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                ],
                device_calibration,
                key,
            )
            if hand_tracking_result.right_hand.wrist_and_palm_normal_device is not None:
                right_wrist_normal_tip = get_point_reprojection(
                    hand_tracking_result.right_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.WRIST)
                    ]
                    + hand_tracking_result.right_hand.wrist_and_palm_normal_device.wrist_normal_device
                    * NORMAL_VIS_LEN,
                    device_calibration,
                    key,
                )
                right_palm_normal_tip = get_point_reprojection(
                    hand_tracking_result.right_hand.landmark_positions_device[
                        int(mps.hand_tracking.HandLandmark.PALM_CENTER)
                    ]
                    + hand_tracking_result.right_hand.wrist_and_palm_normal_device.palm_normal_device
                    * NORMAL_VIS_LEN,
                    device_calibration,
                    key,
                )

        return (
            left_wrist,
            left_palm,
            right_wrist,
            right_palm,
            left_wrist_normal_tip,
            left_palm_normal_tip,
            right_wrist_normal_tip,
            right_palm_normal_tip,
            left_landmarks,
            right_landmarks
        )
    
    def plot_landmarks_and_connections_cv2(
            viz_frame,
            left_landmarks,
            right_landmarks,
            connections,
            img_height
        ):

            viz_frame = viz_frame.copy()
            viz_frame = viz_frame.astype(np.uint8)

            # Colors in BGR format for cv2
            blue_color = (255, 0, 0)  # Blue in BGR
            red_color = (0, 0, 255)   # Red in BGR

            if left_landmarks:
                for left_landmark in left_landmarks:
                    if left_landmark is not None:
                        # import ipdb; ipdb.set_trace()
                        cv2.circle(viz_frame, (int(img_height - 0.5 - left_landmark[1]), int(left_landmark[0] + 0.5)), 5, blue_color, -1)
                for connection in connections:
                    if left_landmarks[int(connection[0])] is not None and left_landmarks[int(connection[1])] is not None:
                        start_point = left_landmarks[int(connection[0])]
                        end_point = left_landmarks[int(connection[1])]
                        cv2.line(viz_frame,
                            (int(img_height - 0.5 - start_point[1]), int(start_point[0] + 0.5)),
                            (int(img_height - 0.5 - end_point[1]), int(end_point[0] + 0.5)),
                            blue_color, 2
                        )

            if right_landmarks:
                for right_landmark in right_landmarks:
                    if right_landmark is not None:
                        cv2.circle(viz_frame, (int(img_height - 0.5 - right_landmark[1]), int(right_landmark[0] + 0.5)), 5, red_color, -1)
                for connection in connections:
                    if right_landmarks[int(connection[0])] is not None and right_landmarks[int(connection[1])] is not None:
                        start_point = right_landmarks[int(connection[0])]
                        end_point = right_landmarks[int(connection[1])]
                        cv2.line(viz_frame,
                            (int(img_height - 0.5 - start_point[1]), int(start_point[0] + 0.5)),
                            (int(img_height - 0.5 - end_point[1]), int(end_point[0] + 0.5)),
                            red_color, 2
                        )

            return viz_frame

    # ================================================
    # Begin main function
    # ================================================

    all_viz_frame_rgb = []
    both_hands_available_list = []

    for rgb_idx in trange(len(stream_timestamps_ns["rgb"])):
        # Get a sample frame for each of the RGB, SLAM left, and SLAM right streams
        sample_timestamp_ns: int = stream_timestamps_ns["rgb"][rgb_idx]
        sample_frames = {
            key: provider.get_image_data_by_time_ns(
                stream_id, sample_timestamp_ns, time_domain, time_query_closest
            )[0]
            for key, stream_id in stream_ids.items()
        }

        # Get the hand tracking pose
        hand_tracking_result = mps_data_provider.get_hand_tracking_result(
            sample_timestamp_ns, time_query_closest
        )

        rgb_image = sample_frames["rgb"].to_numpy_array()
        viz_frame_rgb = np.rot90(rgb_image, -1)
        # print(rgb_image.shape)
        # plt.grid(False)
        # plt.axis("off")
        # plt.imshow(np.rot90(rgb_image, -1))

        (
            left_wrist,
            left_palm,
            right_wrist,
            right_palm,
            left_wrist_normal,
            left_palm_normal,
            right_wrist_normal,
            right_palm_normal,
            left_landmarks,
            right_landmarks,
        ) = get_landmark_pixels("rgb", hand_tracking_result, device_calibration)

        viz_frame_rgb = plot_landmarks_and_connections_cv2(
            viz_frame_rgb,
            left_landmarks,
            right_landmarks,
            mps.hand_tracking.kHandJointConnections,
            viz_frame_rgb.shape[0]
        )

        LEFT_AVAILABLE = hand_tracking_result.left_hand is not None
        RIGHT_AVAILABLE = hand_tracking_result.right_hand is not None

        # Add text indicators for hand availability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Left hand status (blue text to match left hand color)
        left_text = f"Left Hand: {'Available' if LEFT_AVAILABLE else 'Not Available'}"
        left_color = (255, 0, 0) if LEFT_AVAILABLE else (128, 128, 128)  # Blue if available, gray if not
        cv2.putText(viz_frame_rgb, left_text, (10, 30), font, font_scale, left_color, thickness)
        
        # Right hand status (red text to match right hand color)
        right_text = f"Right Hand: {'Available' if RIGHT_AVAILABLE else 'Not Available'}"
        right_color = (0, 0, 255) if RIGHT_AVAILABLE else (128, 128, 128)  # Red if available, gray if not
        cv2.putText(viz_frame_rgb, right_text, (10, 60), font, font_scale, right_color, thickness)

        all_viz_frame_rgb.append(viz_frame_rgb)
        both_hands_available_list.append(LEFT_AVAILABLE and RIGHT_AVAILABLE)

    return all_viz_frame_rgb, both_hands_available_list

def get_episode_idx(both_hands_available_list, sliding_window_length=40):
    cur_episode_idx = -1
    episode_idx_list = []
    for i in range(len(both_hands_available_list)):
        frame_start_idx = max(0, i - sliding_window_length // 2)
        frame_end_idx = min(len(both_hands_available_list), i + sliding_window_length // 2)
        if np.all(both_hands_available_list[frame_start_idx:frame_end_idx]):
            # In a valid frame
            if len(episode_idx_list) == 0 or episode_idx_list[-1] == -1:
                cur_episode_idx += 1
            episode_idx_list.append(cur_episode_idx)
        else:
            episode_idx_list.append(-1)
    return episode_idx_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Aria VRS file and generate hand tracking visualization')
    parser.add_argument('--vrs_sample_path', type=str, help='Path to the VRS sample file')
    parser.add_argument('--output', '-o', type=str, default='output.mp4', help='Output video file path (default: output.mp4)')
    
    args = parser.parse_args()
    
    ret_all_frame, both_hands_available_list = main(args.vrs_sample_path)
    episode_idx_list = get_episode_idx(both_hands_available_list)

    # Write to mp4
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'),
                             30, (ret_all_frame[0].shape[1], ret_all_frame[0].shape[0]))
    for frame_idx, frame in enumerate(ret_all_frame):
        frame_bgr = frame[:, :, ::-1].copy()
        
        # Add episode number text
        episode_num = episode_idx_list[frame_idx]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        episode_text = f"Episode: {episode_num}"
        text_color = (0, 255, 0)  # Green for valid episodes
        
        cv2.putText(frame_bgr, episode_text, (10, 100), font, font_scale, text_color, thickness)
        
        writer.write(frame_bgr)
    writer.release()
