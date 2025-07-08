import os
import cv2
import time
import csv
import clr
import psutil
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
clr.AddReference("OpenHardwareMonitorLib")

from OpenHardwareMonitor import Hardware

computer = Hardware.Computer()
computer.GPUEnabled = True
computer.Open()

def get_amd_gpu_usage():
    for hw in computer.Hardware:
        if hw.HardwareType == Hardware.HardwareType.GpuAti:
            hw.Update()
            for sensor in hw.Sensors:
                if sensor.SensorType == Hardware.SensorType.Load and "Core" in sensor.Name:
                    return sensor.Value 
    return 0.0


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


process = psutil.Process(os.getpid())

for model_file in model_files:
    model_name = os.path.basename(model_file)
    print(f"\n=== Testing {model_name} ===")

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

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{model_name}_performance_log.csv')
    csv_file = open(log_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'FPS', 'CPU (%)', 'RAM (MB)', 'GPU (%)', 'Processed Frames', 'Total Frames'])

    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_FILE}.")
        continue
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / video_fps
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    process.cpu_percent(interval=None)

    print(f"Processing full video...")

    try:
        while True:
            loop_start = time.time()
            frame_count += 1

            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break

            processed_count += 1
            landmarks = extract_landmarks(frame)
            if landmarks is not None:
                input_data = landmarks.reshape(1, -1)

                if model_type == 'sklearn':
                    prediction = model.predict(input_data)
                    predicted_class = str(prediction[0])
                elif model_type == 'catboost':
                    prediction = model.predict(input_data)
                    predicted_class = str(int(prediction[0]))
                elif model_type == 'tensorflow':
                    prediction = model.predict(input_data, verbose=0)
                    predicted_class = str(np.argmax(prediction))
                else:
                    predicted_class = 'N/A'

                cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Hand Detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(f'{model_file}', frame)

            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cpu = process.cpu_percent(interval=None) / psutil.cpu_count(logical=True)
                ram = process.memory_info().rss / (1024 ** 2)
                gpu = get_amd_gpu_usage()
                timestamp = time.strftime('%H:%M:%S', time.localtime())

                print(f'{timestamp} | FPS: {fps:.2f} | CPU: {cpu:.1f}% | RAM: {ram:.1f} MB | GPU: {gpu:.1f}% | Processed: {processed_count}/{frame_count}')
                csv_writer.writerow([timestamp, f"{fps:.2f}", f"{cpu:.1f}", f"{ram:.1f}", f"{gpu:.1f}", f"{processed_count}", f"{frame_count}"])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

            loop_end = time.time()
            elapsed = loop_end - loop_start
            sleep_time = max(0.0, frame_interval - elapsed)
            time.sleep(sleep_time)

    finally:
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()

hands_detector.close()
computer.Close()
print("\nAll models processed on video.")
