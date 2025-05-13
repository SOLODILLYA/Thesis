import os
import cv2

# Settings
input_folder = './from_thesis/one'
output_folder = './from_thesis/one'
filename_prefix = '27_'  # Only files starting with this prefix will be rotated
rotation_angle = 90     # Rotation angle in degrees

# Prepare output folder
os.makedirs(output_folder, exist_ok=True)

# Process files in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and filename.startswith(filename_prefix):
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)

        if image is None:
            print(f"Failed to load {filename}")
            continue

        # Rotate image
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # For 90° clockwise
        # rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)  # For 90° counterclockwise

        # Save to output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, rotated_image)
        print(f"Rotated and saved {output_path}")

print("Done.")
