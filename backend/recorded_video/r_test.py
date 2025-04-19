import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import math
from collections import Counter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                    min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

model = load_model('recorded_video/r_gait_model.h5')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

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

def extract_keypoints(landmarks, image_shape, prev_keypoints=None):
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

def predict_person_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")
    
    predictions = []
    frame_skip = 5
    frame_count = 0
    prev_keypoints = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # if frame_count % frame_skip == 0:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                keypoints = extract_keypoints(results.pose_landmarks.landmark, frame.shape, prev_keypoints)
                features = np.array([[keypoints['height'], keypoints['shoulder_width'], keypoints['hip_width'], 
                                        keypoints['left_knee_angle'], keypoints['right_knee_angle'], 
                                        keypoints['left_elbow_angle'], keypoints['right_elbow_angle'], 
                                        keypoints['height_width_ratio'], keypoints['shoulder_to_hip_ratio']]])
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_name = label_encoder.inverse_transform(predicted_class)
                predictions.append(predicted_name[0])
                prev_keypoints = keypoints
            else:
                print(f"No landmarks found in frame {frame_count}.")
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        # frame_count += 1

    cap.release()

    if predictions:
        most_common_person = Counter(predictions).most_common(1)
        if most_common_person:
            return most_common_person[0][0]  
    return "Unknown"
  
