import os
import cv2
import mediapipe as mp

# Folder with your images
IMAGE_FOLDER = './dataset_created/rock'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Process each image
deleted_count = 0
total_count = 0

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    total_count += 1
    filepath = os.path.join(IMAGE_FOLDER, filename)
    image = cv2.imread(filepath)
    if image is None:
        print(f"Skipping unreadable file: {filename}")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if not result.multi_hand_landmarks:
        os.remove(filepath)
        print(f"Deleted: {filename} (no hand detected)")
        deleted_count += 1

print(f"\nDone. {deleted_count} of {total_count} images deleted.")
