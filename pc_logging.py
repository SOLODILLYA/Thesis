import os
import cv2
import time
import csv
import clr
import psutil
import GPUtil
import numpy as np
import sys
import joblib

# Optional: for landmark extraction (if needed)
import mediapipe as mp

# Path to OpenHardwareMonitor DLL
OPEN_HW_MONITOR_DLL = r"C:/Users/solod/Downloads/openhardwaremonitor-v0.9.6/OpenHardwareMonitor/OpenHardwareMonitorLib.dll"  # <-- adjust this path
sys.path.append(os.path.dirname(OPEN_HW_MONITOR_DLL))
clr.AddReference("OpenHardwareMonitorLib")
from OpenHardwareMonitor import Hardware

# Initialize GPU monitor
computer = Hardware.Computer()
computer.GPUEnabled = True
computer.Open()

def get_amd_gpu_usage():
    for hw in computer.Hardware:
        if hw.HardwareType == Hardware.HardwareType.GpuAti:
            hw.Update()
            for sensor in hw.Sensors:
                if sensor.SensorType == Hardware.SensorType.Load and "Core" in sensor.Name:
                    return sensor.Value  # %
    return 0.0

# List of models to test
model_files = [
    'knn_landmark_model.joblib',
]

# Video file path (replace with your recorded video file)
VIDEO_FILE = 'output.avi'

# Setup MediaPipe (if using landmarks)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Helper: extract landmarks
def extract_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_image)
    if result.multi_hand_landmarks:
        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    return None

# Process info
process = psutil.Process(os.getpid())

for model_file in model_files:
    print(f"\n=== Testing {model_file} ===")
    model = joblib.load(model_file)

    # Prepare log file
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{model_file}_performance_log.csv')
    csv_file = open(log_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'FPS', 'CPU (%)', 'RAM (MB)', 'GPU (%)'])

    # Load video
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_FILE}.")
        continue

    frame_count = 0
    start_time = time.time()
    print(f"Processing full video...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video.")
                break

            frame_count += 1

            landmarks = extract_landmarks(frame)
            if landmarks is not None:
                input_data = landmarks.reshape(1, -1)
                prediction = model.predict(input_data)
                predicted_class = str(prediction[0])
                cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Hand Detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Every 10 frames, log stats
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                cpu = process.cpu_percent(interval=None) / psutil.cpu_count(logical=True)
                ram = process.memory_info().rss / (1024 ** 2)  # MB
                gpu = get_amd_gpu_usage()
                timestamp = time.strftime('%H:%M:%S', time.localtime())

                print(f'{timestamp} | FPS: {fps:.2f} | CPU: {cpu:.1f}% | RAM: {ram:.1f} MB | GPU: {gpu:.1f}%')
                csv_writer.writerow([timestamp, f"{fps:.2f}", f"{cpu:.1f}", f"{ram:.1f}", f"{gpu:.1f}"])

            cv2.imshow(f'{model_file}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user.")
                break

    finally:
        cap.release()
        csv_file.close()

cv2.destroyAllWindows()
hands_detector.close()
computer.Close()
print("\nAll models processed on video.")