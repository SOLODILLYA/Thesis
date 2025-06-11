import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib
from sklearn.metrics import f1_score
# Dataset directory
DATASET_DIR = 'dataset_landmarks'
MAX_SAMPLES_PER_CLASS = 2000

X = []
y = []
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

def load_landmarks(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        if not lines or "No hand landmarks" in lines[0]:
            return None
        values = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 4:
                values.extend([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(values)

# Load dataset with random sampling
for label_idx, label in enumerate(class_names):
    label_path = os.path.join(DATASET_DIR, label)
    all_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    sampled_files = random.sample(all_files, min(len(all_files), MAX_SAMPLES_PER_CLASS))

    for file in sampled_files:
        data = load_landmarks(os.path.join(label_path, file))
        if data is not None:
            X.append(data)
            y.append(label_idx)

X = np.array(X)
y = np.array(y)

# Stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create and train Logistic Regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Evaluate
y_pred = logreg.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))

# Confusion matrix with values
cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Confusion Matrix with Counts")
plt.show()

print(f"Validation Accuracy: {accuracy:.4f}")
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Weighted F1 Score: {f1:.4f}")
# Save model
joblib.dump(logreg, 'logistic_regression_model.joblib')
print("Logistic Regression model saved as 'logistic_regression_model.joblib'")
