import cv2
import numpy as np
import joblib
import mediapipe as mp

# Load trained KNN model
model = joblib.load('knn_gesture_model.joblib')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Feature extraction from hand crop
def extract_features_from_crop(crop):
    resized = cv2.resize(crop, (64, 64))
    features = resized.flatten()
    return features.reshape(1, -1)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    label = "No Hand Detected"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
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

            hand_crop = frame[ymin:ymax, xmin:xmax]

            if hand_crop.size != 0:
                features = extract_features_from_crop(hand_crop)
                prediction = model.predict(features)
                label = prediction[0]

            # Draw bounding box and label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {label}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show live frame
    cv2.imshow('Live KNN Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
