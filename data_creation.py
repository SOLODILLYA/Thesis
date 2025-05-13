import cv2
import os
import time

# Parameters
output_folder = './dataset_created/peace'
total_frames_to_capture = 500
capture_interval_sec = 0.05  # 50 ms = 20 FPS approx.

# Prepare output folder
os.makedirs(output_folder, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_counter = 0
last_capture_time = time.time()

print(f"Capturing {total_frames_to_capture} frames. Please wait...")
while frame_counter < total_frames_to_capture:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Show preview
    cv2.imshow('Webcam Preview', frame)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval_sec:
        frame_filename = os.path.join(output_folder, f'frame_{frame_counter:05d}_rock_left_back_1.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        frame_counter += 1
        last_capture_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Interrupted by user.")
        break

print("Capture complete.")
cap.release()
cv2.destroyAllWindows()
