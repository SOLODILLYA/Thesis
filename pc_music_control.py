import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import joblib
import ctypes

# Load trained model (adjust path as needed)
model = joblib.load('lightgbm_model.joblib')
class_names = ['like', 'no_gesture', 'peace', 'rock']  # Adjust to your class names

# Gesture-to-media control mapping
gesture_actions = {
    'like': lambda: pyautogui.press('playpause'),   # Play / Pause
    'rock': lambda: pyautogui.press('prevtrack'),   # Previous Song
    'peace': lambda: pyautogui.press('nexttrack')   # Next Song
}

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def extract_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks), hand_landmarks
    return None, None

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

last_prediction = None
frame_counter = 0

print("Starting Music Control. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    landmarks, hand_landmarks = extract_landmarks(frame)

    if landmarks is not None:
        input_data = landmarks.reshape(1, -1)
        prediction_idx = model.predict(input_data)[0]
        predicted_class = class_names[prediction_idx]

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Gesture: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Trigger action with custom cooldown for 'like'
        required_cooldown = 50
        if predicted_class in gesture_actions:
            if predicted_class != last_prediction or frame_counter > required_cooldown:
                print(f"Triggered: {predicted_class}")
                gesture_actions[predicted_class]()
                last_prediction = predicted_class
                frame_counter = 0
    else:
        cv2.putText(frame, 'No Hand Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    frame_counter += 1
    cv2.imshow('Music Gesture Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
