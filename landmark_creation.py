import cv2
import os
import numpy as np
import mediapipe as mp

# Setup output directory
output_folder = './dataset_landmarks/peace'
os.makedirs(output_folder, exist_ok=True)

# Target number of samples to collect
target_samples = 500
collected_samples = 0

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("Starting data collection. Press 'q' to stop manually if needed.")

while collected_samples < target_samples:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw landmarks on frame for visualization
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks data
        landmark_data = []
        for idx, lm in enumerate(hand_landmarks.landmark):
            landmark_data.append(f"{idx},{lm.x},{lm.y},{lm.z}")

        # Save to file
        output_path = os.path.join(output_folder, f'sample_{collected_samples:04d}_peace_left_3.txt')
        with open(output_path, 'w') as f:
            f.write('\n'.join(landmark_data))

        collected_samples += 1
        print(f"Collected {collected_samples}/{target_samples}")

    # Show frame with landmarks
    cv2.imshow('Collecting Hand Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print(f"Data collection complete. {collected_samples} samples saved in '{output_folder}'.")
