import os
import cv2
import time
import csv
import clr
import psutil
import GPUtil
import numpy as np
import sys
import tensorflow as tf
import mediapipe as mp

# Path to OpenHardwareMonitor DLL
OPEN_HW_MONITOR_DLL = r"C:/Users/solod/Downloads/openhardwaremonitor-v0.9.6/OpenHardwareMonitor/OpenHardwareMonitorLib.dll"
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

# Video file path
VIDEO_FILE = 'output.avi'

# Load TensorFlow model
model_file = 'best_cnn_model_landmarks.h5'
print(f"\n=== Testing {model_file} ===")
model = tf.keras.models.load_model(model_file)

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

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
    sys.exit()

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
            prediction = model.predict(input_data, verbose=0)
            predicted_class = str(np.argmax(prediction))
            cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Hand Detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
    print("\nTensorFlow CNN model processed on video.")
