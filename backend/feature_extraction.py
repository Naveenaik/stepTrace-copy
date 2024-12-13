import cv2
import mediapipe as mp
import pandas as pd
import math
import os
import threading

STOP_SIGNAL = threading.Event()
output_csv = 'live_output_gait_parameters.csv'

def calculate_angle(a, b, c):
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    ab_magnitude = math.sqrt(ab[0]**2 + ab[1]**2)
    bc_magnitude = math.sqrt(bc[0]**2 + bc[1]**2)
    return 0 if ab_magnitude * bc_magnitude == 0 else math.degrees(math.acos(dot_product / (ab_magnitude * bc_magnitude)))

def extract_keypoints(landmarks, image_shape):
    h, w = image_shape[:2]
    keypoints = {}

    # Heights and widths
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

    left_knee_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h
    right_knee_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h
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

def extract_features_from_video(video_path, person_name):
    try:
        while not STOP_SIGNAL.is_set():
            frame_rate = 30
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Error: Unable to open video file {video_path}")

            keypoints_list = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    keypoints = extract_keypoints(landmarks, frame.shape)
                    keypoints['person_name'] = person_name
                    keypoints_list.append(keypoints)

                    df = pd.DataFrame(keypoints_list)
                    if os.path.exists(output_csv):
                        df.to_csv(output_csv, mode='a', header=False, index=False)
                    else:
                        df.to_csv(output_csv, index=False)

            cap.release()

            if not keypoints_list:
                raise Exception(f"No landmarks detected in video {video_path}")

            # df = pd.DataFrame(keypoints_list)
            # if os.path.exists(output_csv):
            #     df.to_csv(output_csv, mode='a', header=False, index=False)
            # else:
            #     df.to_csv(output_csv, index=False)
            # print(f"Features extracted and saved to {output_csv}")
            pass
        print("Feature extraction stopped.")
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        raise e