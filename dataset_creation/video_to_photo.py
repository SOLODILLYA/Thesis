import cv2
import os

for i in range(0, 4):  # From 4 to 27 inclusive
    # Input video path
    video_path = f'./video_dataset/{i}/one.mp4'  # You may adjust 'four.mp4' if filenames vary

    # Output folder
    output_folder = f'./from_thesis/one'
    os.makedirs(output_folder, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        continue  # Skip to next if video not found

    frame_counter = 0
    print(f"Extracting frames from {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_filename = os.path.join(output_folder, f'{i}_frame_{frame_counter:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        frame_counter += 1

    cap.release()
    print("Done. All frames saved in", output_folder)
