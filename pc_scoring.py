import os
import pandas as pd

log_folder = "logs"
output_file = "model_scores_summary.csv"
summary_data = []

w_fps = 2.0
w_cpu = 1.0
w_ram = 0.5
w_gpu = 1.0

def compute_cost(fps, cpu, ram, gpu, ε=1e-5):
    return cpu * ram * gpu / fps

for file in os.listdir(log_folder):
    if file.endswith(".csv"):
        filepath = os.path.join(log_folder, file)
        df = pd.read_csv(filepath)

        required_cols = {'FPS', 'CPU (%)', 'RAM (MB)', 'GPU (%)'}
        if not required_cols.issubset(df.columns):
            continue

        avg_fps = df['FPS'].astype(float).mean()
        avg_cpu = df['CPU (%)'].astype(float).mean()
        avg_ram = df['RAM (MB)'].astype(float).mean()
        avg_gpu = df['GPU (%)'].astype(float).mean()
        score = compute_cost(avg_fps, avg_cpu, avg_ram, avg_gpu)

        summary_data.append({
            'Model': file.replace('_performance_log.csv', '').replace('.csv', ''),
            'Avg FPS': round(avg_fps, 2),
            'Avg CPU (%)': round(avg_cpu, 2),
            'Avg RAM (MB)': round(avg_ram, 2),
            'Avg GPU (%)': round(avg_gpu, 2),
            'Score': round(score, 4)
        })

summary_df = pd.DataFrame(summary_data)
summary_df.sort_values(by='Score', inplace=True)
summary_df.to_csv(output_file, index=False)

print(f"Scoring complete. Summary saved to '{output_file}'.")