import cv2
import numpy as np
import mediapipe as mp
from catboost import CatBoostClassifier

# Load the trained CatBoost model
cat_model = CatBoostClassifier()
cat_model.load_model('catboost_model.cbm')
class_names = ['like', 'no_gesture', 'peace', 'rock']  # Adjust to your classes

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

print("Starting real-time prediction. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    landmarks, hand_landmarks = extract_landmarks(frame)

    if landmarks is not None:
        # Reshape for prediction
        input_data = landmarks.reshape(1, -1)
        prediction = cat_model.predict(input_data)[0]
        predicted_class = class_names[int(prediction)]

        # Draw landmarks and prediction
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No Hand Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
