import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7)
def calculate_angle(a, b, c):
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    ab_magnitude = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    bc_magnitude = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if ab_magnitude * bc_magnitude == 0:
        return 0
    angle = math.acos(dot_product / (ab_magnitude * bc_magnitude))
    return math.degrees(angle)

def extract_keypoints(landmarks, image_shape):
    h, w = image_shape[:2]
    keypoints = {}

    head_y = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y * h
    left_foot_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h
    right_foot_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h
    height = abs(max(left_foot_y, right_foot_y) - head_y)

    left_shoulder_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w
    right_shoulder_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w
    shoulder_width = abs(right_shoulder_x - left_shoulder_x)

    left_hip_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w
    right_hip_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w
    hip_width = abs(right_hip_x - left_hip_x)

    # Add the foot_y values to the keypoints dictionary
    keypoints['left_foot_y'] = left_foot_y
    keypoints['right_foot_y'] = right_foot_y

    left_knee_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h)
    )

    right_knee_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
    )

    left_elbow_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * h)
    )

    right_elbow_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h)
    )

    keypoints['height'] = height
    keypoints['shoulder_width'] = shoulder_width
    keypoints['hip_width'] = hip_width
    keypoints['left_knee_angle'] = left_knee_angle
    keypoints['right_knee_angle'] = right_knee_angle
    keypoints['left_elbow_angle'] = left_elbow_angle
    keypoints['right_elbow_angle'] = right_elbow_angle
    keypoints['height_width_ratio'] = height / shoulder_width if shoulder_width > 0 else 0
    keypoints['shoulder_to_hip_ratio'] = shoulder_width / hip_width if hip_width > 0 else 0

    return keypoints

    h, w = image_shape[:2]
    keypoints = {}

    head_y = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value].y * h
    left_foot_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h
    right_foot_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h
    height = abs(max(left_foot_y, right_foot_y) - head_y)

    left_shoulder_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w
    right_shoulder_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w
    shoulder_width = abs(right_shoulder_x - left_shoulder_x)

    left_hip_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w
    right_hip_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w
    hip_width = abs(right_hip_x - left_hip_x)

    left_knee_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h)
    )

    right_knee_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
    )

    left_elbow_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * h)
    )

    right_elbow_angle = calculate_angle(
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h),
        (landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h)
    )

    keypoints['height'] = height
    keypoints['shoulder_width'] = shoulder_width
    keypoints['hip_width'] = hip_width
    keypoints['left_knee_angle'] = left_knee_angle
    keypoints['right_knee_angle'] = right_knee_angle
    keypoints['left_elbow_angle'] = left_elbow_angle
    keypoints['right_elbow_angle'] = right_elbow_angle
    keypoints['height_width_ratio'] = height / shoulder_width if shoulder_width > 0 else 0
    keypoints['shoulder_to_hip_ratio'] = shoulder_width / hip_width if hip_width > 0 else 0

    return keypoints


def draw_skeleton(image, landmarks):
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=5),
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=5)
    )


def process_video(video_path, person_name, output_dir):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    prev_left_foot_y = prev_right_foot_y = None
    frame_num = 0

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            keypoints = extract_keypoints(results.pose_landmarks.landmark, frame.shape)

            if prev_left_foot_y is not None and prev_right_foot_y is not None:
                keypoints['left_foot_speed'] = abs(keypoints['left_foot_y'] - prev_left_foot_y)
                keypoints['right_foot_speed'] = abs(keypoints['right_foot_y'] - prev_right_foot_y)

            prev_left_foot_y = keypoints['left_foot_y']
            prev_right_foot_y = keypoints['right_foot_y']

            keypoints['frame'] = frame_num
            keypoints['person_name'] = person_name
            keypoints_list.append(keypoints)

            draw_skeleton(frame, results.pose_landmarks)

            frame_filename = os.path.join(output_dir, f"{person_name}_frame_{frame_num}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_num += 1

    cap.release()
    return keypoints_list

def process_videos_in_directory(directory_path, output_csv, output_frames_dir):
    all_keypoints = []      

    for video_file in os.listdir(directory_path):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(directory_path, video_file)
            person_name = os.path.splitext(video_file)[0]
            print(f"Processing {video_file}...")
            person_frames_dir = os.path.join(output_frames_dir, person_name)
            keypoints_list = process_video(video_path, person_name, person_frames_dir)
            if keypoints_list:
                all_keypoints.extend(keypoints_list)

    df = pd.DataFrame(all_keypoints)
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")


video_directory = 'data' 
output_csv = 'r_output_gait_parameters_per_frame.csv' 
output_frames_dir = 'skeleton_frames' 

process_videos_in_directory(video_directory, output_csv, output_frames_dir)
