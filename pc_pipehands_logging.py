import os
import sys
import clr
import cv2
import time
import csv
import psutil
import GPUtil
import numpy as np
import mediapipe as mp

# Load machine-specific DLL path from local config
try:
    from config_local import OPEN_HW_MONITOR_DLL
except ImportError:
    raise ImportError("⚠ Missing config_local.py! Please create it and set OPEN_HW_MONITOR_DLL.")

# Append DLL path directory
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

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def detect_hand(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_image)
    return result.multi_hand_landmarks is not None

# Process info
process = psutil.Process(os.getpid())

# Prepare log file
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'hand_detection_performance_log.csv')
csv_file = open(log_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'FPS', 'CPU (%)', 'RAM (MB)', 'GPU (%)', 'Hand Detected'])

# Load video
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"Error: Cannot open video {VIDEO_FILE}.")
    sys.exit()

frame_count = 0
start_time = time.time()

# Warm up CPU tracker
process.cpu_percent(interval=None)

print("Processing full video for hand detection...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video.")
            break

        frame_count += 1

        hand_detected = detect_hand(frame)
        label = 'Hand Detected' if hand_detected else 'No Hand'
        color = (0, 255, 0) if hand_detected else (0, 0, 255)

        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Hand Detection', frame)

        # Log stats every 10 frames
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            cpu = process.cpu_percent(interval=None) / psutil.cpu_count(logical=True)
            ram = process.memory_info().rss / (1024 ** 2)  # MB
            gpu = get_amd_gpu_usage()
            timestamp = time.strftime('%H:%M:%S', time.localtime())

            print(f'{timestamp} | FPS: {fps:.2f} | CPU: {cpu:.1f}% | RAM: {ram:.1f} MB | GPU: {gpu:.1f}% | {label}')
            csv_writer.writerow([timestamp, f"{fps:.2f}", f"{cpu:.1f}", f"{ram:.1f}", f"{gpu:.1f}", label])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    hands_detector.close()
    computer.Close()
    print("\nHand detection script finished.")
