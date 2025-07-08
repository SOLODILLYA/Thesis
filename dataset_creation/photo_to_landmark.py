import os
import cv2
import mediapipe as mp

input_folder = './dataset_created/peace' 
output_folder = './dataset_landmarks/peace'
os.makedirs(output_folder, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load {img_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    landmark_data = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                landmark_data.append(f"{id},{lm.x},{lm.y},{lm.z}")

    output_path = os.path.join(output_folder, img_name.rsplit('.', 1)[0] + '.txt')
    with open(output_path, 'w') as f:
        if landmark_data:
            f.write('\n'.join(landmark_data))

    print(f"Processed {img_name}")

hands.close()
print("Processing complete.")
