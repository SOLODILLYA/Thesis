import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
import sys

from tensorflow.keras.models import load_model
from catboost import CatBoostClassifier

# All available models
model_files = [
    'models/bagging_model.joblib',
    'hist_gradient_boost_model.joblib',
    'knn_landmark_model.joblib',
    'lightgbm_model.joblib',
    'logistic_regression_model.joblib',
    'mlp_model.joblib',
    'random_forest_model.joblib',
    'svm_model.joblib',
    'xgboost_model.joblib',
    'catboost_model.cbm',
    'best_cnn_model_landmarks.h5'
]

# Change or prompt for model selection
print("Select model to load:")
for i, f in enumerate(model_files):
    print(f"{i}: {f}")

choice = int(input("Enter number: "))
model_path = model_files[choice]

# Load selected model
if model_path.endswith('.joblib'):
    model = joblib.load(model_path)
    def predict_fn(X): return model.predict(X)[0]

elif model_path.endswith('.cbm'):
    model = CatBoostClassifier()
    model.load_model(model_path)
    def predict_fn(X): return model.predict(X)[0]

elif model_path.endswith('.h5'):
    model = load_model(model_path)
    def predict_fn(X): return np.argmax(model.predict(X), axis=1)[0]

else:
    print("Unsupported model type.")
    sys.exit(1)

# Class labels
class_names = ['like', 'no_gesture', 'peace', 'rock']

# MediaPipe setup
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

print(f"Running real-time prediction with model: {model_path}")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    landmarks, hand_landmarks = extract_landmarks(frame)

    if landmarks is not None:
        input_data = landmarks.reshape(1, -1)
        prediction = predict_fn(input_data)
        predicted_class = class_names[int(prediction)]

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No Hand Detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
