import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load validation data
X_val = np.load('../X_val.npy')
y_val = np.load('../y_val.npy')

# Load Booster model
bst = joblib.load('../lightgbm_model_full.joblib')

# Get best iteration or fallback to full
best_iter = getattr(bst, 'best_iteration', None)
if best_iter is None or best_iter == 0:
    best_iter = bst.current_iteration()
print(f"Pruning to {best_iter} boosting rounds...")

# Predict and evaluate
y_pred_prob = bst.predict(X_val, num_iteration=best_iter)
y_pred = np.argmax(y_pred_prob, axis=1)

accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')

print(f"Pruned Model Accuracy: {accuracy:.4f}")
print(f"Pruned Weighted F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save pruned Booster
joblib.dump(bst, '../lightgbm_model_pruned.joblib')
print("Pruned Booster saved as 'lightgbm_model_pruned.joblib'")
