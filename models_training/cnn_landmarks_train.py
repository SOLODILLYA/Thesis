import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score

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
            if len(parts) == 4:  # id, x, y, z
                values.extend([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(values)

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

# Stratified split for balanced validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Build Stronger Model with Dropout
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Custom callback to log best validation accuracy
class BestModelLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

    def on_train_end(self, logs=None):
        print(f"\nBest validation accuracy achieved: {self.best_val_acc:.4f}")

# Setup checkpoint to save best model only
checkpoint = ModelCheckpoint(
    'best_cnn_model_landmarks.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

best_model_logger = BestModelLogger()

# Train model with checkpoint and logger
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[checkpoint, best_model_logger]
)

# Plot training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# Load and evaluate best saved model on validation data
print("\nEvaluating the best saved model on the validation set...")
best_model = tf.keras.models.load_model('best_cnn_model_landmarks.h5')
val_loss, val_accuracy = best_model.evaluate(val_ds)
print(f"Validation accuracy of the saved best model: {val_accuracy:.4f}")

# Detailed prediction report
y_pred_probs = best_model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Weighted F1 Score: {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred, target_names=class_names))
