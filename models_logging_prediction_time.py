import os
import cv2
import time
import csv
import numpy as np
import sys
import joblib
import mediapipe as mp
from catboost import CatBoostClassifier
import tensorflow as tf

try:
    from config_local import OPEN_HW_MONITOR_DLL
except ImportError:
    raise ImportError("Missing config_local.py! Please create it and set OPEN_HW_MONITOR_DLL.")

sys.path.append(os.path.dirname(OPEN_HW_MONITOR_DLL))
import clr
clr.AddReference("OpenHardwareMonitorLib")
from OpenHardwareMonitor import Hardware

model_dir = 'models_test'
model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

VIDEO_FILE = 'dataset_creation/output.avi'

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def extract_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_image)
    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)
    return None

log_dir = 'logs_speed'
os.makedirs(log_dir, exist_ok=True)

for model_file in model_files:
    model_name = os.path.basename(model_file)
    print(f"\n=== Testing Prediction Time: {model_name} ===")

    if model_name.endswith('.joblib'):
        model = joblib.load(model_file)
        model_type = 'sklearn'
    elif model_name.endswith('.cbm'):
        model = CatBoostClassifier()
        model.load_model(model_file)
        model_type = 'catboost'
    elif model_name.endswith('.h5'):
        model = tf.keras.models.load_model(model_file)
        model_type = 'tensorflow'
    else:
        print(f"Unsupported model type for {model_name}")
        continue

    log_path = os.path.join(log_dir, f'{model_name}_prediction_time.csv')
    csv_file = open(log_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Prediction Time (ms)'])

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_FILE}.")
        csv_file.close()
        continue

    frame_num = 0
    print("Measuring prediction time only...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            input_data = landmarks.reshape(1, -1)

            start_pred = time.time()
            if model_type == 'sklearn':
                _ = model.predict(input_data)
            elif model_type == 'catboost':
                _ = model.predict(input_data)
            elif model_type == 'tensorflow':
                _ = model.predict(input_data, verbose=0)
            pred_time = (time.time() - start_pred) * 1000

            frame_num += 1
            csv_writer.writerow([frame_num, f"{pred_time:.2f}"])

            if frame_num % 10 == 0:
                print(f"Frame {frame_num} | Prediction Time: {pred_time:.2f} ms")

    cap.release()
    csv_file.close()
    print(f"Finished: {model_name}")

hands_detector.close()
print("\nAll models prediction time measured.")
