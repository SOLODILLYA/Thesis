import os
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Set dataset directory
DATASET_DIR = './dataset_created/'

# Prepare data containers
X = []
y = []

# Helper to crop hand region from an image
def crop_hand_area(image):
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')

    rgb_image = np.array(image)  # Already in RGB
    result = hands.process(rgb_image)
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        h, w, _ = rgb_image.shape
        x_coords = [int(lm.x * w) for lm in landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in landmarks.landmark]
        xmin, xmax = max(0, min(x_coords)), min(w, max(x_coords))
        ymin, ymax = max(0, min(y_coords)), min(h, max(y_coords))
        margin_x = int((xmax - xmin) * 0.2)
        margin_y = int((ymax - ymin) * 0.2)
        xmin = max(0, xmin - margin_x)
        xmax = min(w, xmax + margin_x)
        ymin = max(0, ymin - margin_y)
        ymax = min(h, ymax + margin_y)
        hand_crop = rgb_image[ymin:ymax, xmin:xmax]
        if hand_crop.size == 0:
            return None
        return Image.fromarray(hand_crop)  # Keep RGB
    return None

# Load and preprocess images
for gesture in os.listdir(DATASET_DIR):
    gesture_path = os.path.join(DATASET_DIR, gesture)
    if not os.path.isdir(gesture_path):
        continue

    for image_file in os.listdir(gesture_path):
        image_path = os.path.join(gesture_path, image_file)
        try:
            img = Image.open(image_path).convert('RGB')  # Keep RGB
            hand_crop = crop_hand_area(img)

            if hand_crop is not None:
                hand_crop = hand_crop.resize((64, 64))
                img_array = np.array(hand_crop).flatten()  # Keep RGB, no grayscale conversion
                X.append(img_array)
                y.append(gesture)
            else:
                print(f"No hand detected in {image_path}, skipping.")
        except Exception as e:
            print(f"Skipping {image_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

print("\nPer-Class Accuracy:")
for i, label in enumerate(knn.classes_):
    true_positives = cm[i, i]
    total_samples = cm[i].sum()
    class_accuracy = true_positives / total_samples if total_samples > 0 else 0
    print(f"{label}: {class_accuracy:.2f}")

# Save model
joblib.dump(knn, 'knn_gesture_model.joblib')
