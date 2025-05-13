import os
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# Input and output folders
input_folder = 'input_photos'
output_folder = 'cropped_photos'
os.makedirs(output_folder, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    input_path = os.path.join(input_folder, filename)
    image_bgr = cv2.imread(input_path)
    if image_bgr is None:
        print(f"Failed to load {filename}")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        print(f"No hand detected in {filename}, skipping.")
        continue

    # Extract bounding box with margin
    hand_landmarks = result.multi_hand_landmarks[0]
    h, w, _ = image_rgb.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    xmin, xmax = max(0, min(x_coords)), min(w, max(x_coords))
    ymin, ymax = max(0, min(y_coords)), min(h, max(y_coords))
    margin_x = int((xmax - xmin) * 0.2)
    margin_y = int((ymax - ymin) * 0.2)
    xmin = max(0, xmin - margin_x)
    xmax = min(w, xmax + margin_x)
    ymin = max(0, ymin - margin_y)
    ymax = min(h, ymax + margin_y)

    hand_crop = image_bgr[ymin:ymax, xmin:xmax]
    if hand_crop.size == 0:
        print(f"Empty crop in {filename}, skipping.")
        continue

    # Save cropped image
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, hand_crop)
    print(f"Cropped and saved {output_path}")

print("Batch cropping completed.")
