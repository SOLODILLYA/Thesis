import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import f1_score
# Directory with subfolders per gesture, containing landmark .txt files
DATASET_DIR = '../dataset_landmarks/'

# Prepare data containers
X = []
y = []

def load_landmarks(txt_file_path):
    try:
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
            if not lines or lines[0].startswith("No hand landmarks detected"):
                return None
            landmarks = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 4:  # id,x,y,z
                    landmarks.extend([float(parts[1]), float(parts[2]), float(parts[3])])
            return np.array(landmarks)
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        return None

# Load data
for gesture in os.listdir(DATASET_DIR):
    gesture_path = os.path.join(DATASET_DIR, gesture)
    if not os.path.isdir(gesture_path):
        continue

    for txt_file in os.listdir(gesture_path):
        if not txt_file.endswith('.txt'):
            continue
        txt_file_path = os.path.join(gesture_path, txt_file)
        landmarks = load_landmarks(txt_file_path)
        if landmarks is not None:
            X.append(landmarks)
            y.append(gesture)
        else:
            print(f"Skipping {txt_file_path} due to invalid or missing landmarks.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("No valid data found. Exiting.")
    exit()

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
print(f"Overall Accuracy: {accuracy:.4f}")

print("\nPer-Class Accuracy:")
for i, label in enumerate(knn.classes_):
    true_positives = cm[i, i]
    total_samples = cm[i].sum()
    class_accuracy = true_positives / total_samples if total_samples > 0 else 0
    print(f"{label}: {class_accuracy:.2f}")
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Weighted F1 Score: {f1:.4f}")
# Save model
joblib.dump(knn, 'knn_landmark_model.joblib')
print("Model saved as knn_landmark_model.joblib")
