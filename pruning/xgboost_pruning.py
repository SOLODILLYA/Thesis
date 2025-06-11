import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load saved data (adjust paths if needed)
X_val = np.load('../X_val.npy')
y_val = np.load('../y_val.npy')

# Load full model
model_path = '../xgboost_model_full.joblib'
xgb_model = joblib.load(model_path)

# Check if best_iteration is available
if hasattr(xgb_model, 'best_iteration') and xgb_model.best_iteration is not None:
    best_iter = xgb_model.best_iteration + 1
    print(f"Pruning model to {best_iter} estimators based on early stopping")
    xgb_model.set_params(n_estimators=best_iter)
    
    # Evaluate on validation data
    y_pred = xgb_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"Pruned Model Accuracy: {accuracy:.4f}")
    print(f"Pruned Weighted F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Save pruned model
    pruned_model_path = '../xgboost_model_pruned.joblib'
    joblib.dump(xgb_model, pruned_model_path)
    print(f"Pruned XGBoost model saved as '{pruned_model_path}'")
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")

else:
    print("No best_iteration found in the model")
