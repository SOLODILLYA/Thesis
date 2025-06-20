import os
import pandas as pd

# Folder containing speed logs
log_folder = "logs_speed"
output_file = "model_speed_summary.csv"
summary_data = []

# Scan and process each prediction time CSV
for file in os.listdir(log_folder):
    if file.endswith(".csv"):
        filepath = os.path.join(log_folder, file)
        df = pd.read_csv(filepath)

        # Ensure required column exists
        if 'Prediction Time (ms)' not in df.columns:
            continue

        # Compute average prediction time (ms)
        avg_pred_time = df['Prediction Time (ms)'].astype(float).mean()

        summary_data.append({
            'Model': file.replace('_prediction_time.csv', '').replace('.csv', ''),
            'Avg Prediction Time (ms)': round(avg_pred_time, 2)
        })

# Create summary CSV
summary_df = pd.DataFrame(summary_data)
summary_df.sort_values(by='Avg Prediction Time (ms)', inplace=True)
summary_df.to_csv(output_file, index=False)

print(f"Speed scoring complete. Summary saved to '{output_file}'.")
