import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Landmark indices from MediaPipe documentation
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Helper function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Helper function to calculate angle between three points (p2 is the vertex)
def calculate_angle(p1, p2, p3):
    v1_x = p1.x - p2.x
    v1_y = p1.y - p2.y
    v2_x = p3.x - p2.x
    v2_y = p3.y - p2.y

    dot_product = v1_x * v2_x + v1_y * v2_y
    mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
    mag_v2 = math.sqrt(v2_x**2 + v2_y**2)

    if mag_v1 * mag_v2 == 0:
        return 180.0 # Return a neutral angle if magnitude is zero to avoid errors
    
    acos_arg = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
    angle_rad = math.acos(acos_arg)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def is_thumbs_up(landmarks, debug=False):
    angle_thumb_mcp = calculate_angle(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_IP])
    angle_thumb_ip = calculate_angle(landmarks[THUMB_MCP], landmarks[THUMB_IP], landmarks[THUMB_TIP])
    thumb_extended = angle_thumb_mcp > 130 and angle_thumb_ip > 130

    thumb_tip_y = landmarks[THUMB_TIP].y
    thumb_ip_y = landmarks[THUMB_IP].y
    thumb_mcp_y = landmarks[THUMB_MCP].y
    wrist_y = landmarks[WRIST].y
    
    thumb_is_up = (thumb_tip_y < thumb_ip_y and 
                   thumb_tip_y < thumb_mcp_y and 
                   thumb_tip_y < wrist_y * 0.95)

    other_fingers_mcp_avg_y = (
        landmarks[INDEX_FINGER_MCP].y +
        landmarks[MIDDLE_FINGER_MCP].y +
        landmarks[RING_FINGER_MCP].y +
        landmarks[PINKY_MCP].y) / 4.0
    thumb_is_up = thumb_is_up and (thumb_tip_y < other_fingers_mcp_avg_y)

    flexed_finger_count = 0
    finger_names = ["Index", "Middle", "Ring", "Pinky"]
    finger_mcp_joints = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
    finger_pip_joints = [INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP]
    finger_dip_joints = [INDEX_FINGER_DIP, MIDDLE_FINGER_DIP, RING_FINGER_DIP, PINKY_DIP]
    finger_tip_joints = [INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]

    for i in range(4):
        mcp = landmarks[finger_mcp_joints[i]]
        pip = landmarks[finger_pip_joints[i]]
        dip = landmarks[finger_dip_joints[i]]
        tip = landmarks[finger_tip_joints[i]]

        angle_pip = calculate_angle(mcp, pip, dip)
        angle_dip = calculate_angle(pip, dip, tip)
        
        pip_flex_threshold = 170 
        
        finger_is_flexed_this_iteration = False
        if angle_pip < pip_flex_threshold: 
            if angle_pip < 130: 
                finger_is_flexed_this_iteration = (tip.y > mcp.y * 0.90) 
            else: 
                finger_is_flexed_this_iteration = (angle_dip < 178 and tip.y > mcp.y * 0.90) 
        
        if finger_is_flexed_this_iteration:
            flexed_finger_count += 1
        elif debug:
            print(f"ThumbsUp V10 - {finger_names[i]} not counted as flexed: angle_pip={angle_pip:.1f} (Thresh: <{pip_flex_threshold}), angle_dip={angle_dip:.1f}, tip.y={tip.y:.2f}, mcp.y*0.90={(mcp.y * 0.90):.2f}")

    fingers_flexed_majority = flexed_finger_count >= 3

    if debug:
        print(f"ThumbsUp V10 - Thumb Extended: {thumb_extended} (MCP Angle: {angle_thumb_mcp:.1f}, IP Angle: {angle_thumb_ip:.1f})")
        print(f"ThumbsUp V10 - Thumb Is Up: {thumb_is_up} (TipY: {thumb_tip_y:.2f}, WristY: {wrist_y:.2f}, OtherMCP_AvgY: {other_fingers_mcp_avg_y:.2f})")
        print(f"ThumbsUp V10 - Flexed Finger Count: {flexed_finger_count} (Required >= 3 for majority)")
            
    return thumb_extended and thumb_is_up and fingers_flexed_majority

def is_peace_sign(landmarks, debug=False):
    # 1. Index Finger Extension
    angle_index_pip = calculate_angle(landmarks[INDEX_FINGER_MCP], landmarks[INDEX_FINGER_PIP], landmarks[INDEX_FINGER_DIP])
    angle_index_dip = calculate_angle(landmarks[INDEX_FINGER_PIP], landmarks[INDEX_FINGER_DIP], landmarks[INDEX_FINGER_TIP])
    index_extended = angle_index_pip > 150 and angle_index_dip > 150

    # 2. Middle Finger Extension
    angle_middle_pip = calculate_angle(landmarks[MIDDLE_FINGER_MCP], landmarks[MIDDLE_FINGER_PIP], landmarks[MIDDLE_FINGER_DIP])
    angle_middle_dip = calculate_angle(landmarks[MIDDLE_FINGER_PIP], landmarks[MIDDLE_FINGER_DIP], landmarks[MIDDLE_FINGER_TIP])
    middle_extended = angle_middle_pip > 150 and angle_middle_dip > 150

    # 3. Ring Finger Flexion - Must be flexed
    angle_ring_pip = calculate_angle(landmarks[RING_FINGER_MCP], landmarks[RING_FINGER_PIP], landmarks[RING_FINGER_DIP])
    ring_flexed = False
    if angle_ring_pip < 90: # Strongly flexed
        ring_flexed = True
    elif angle_ring_pip < 130: # Moderately flexed, check y-pos loosely
        ring_flexed = landmarks[RING_FINGER_TIP].y > landmarks[RING_FINGER_MCP].y * 0.98

    # 4. Pinky Finger Flexion - Must be flexed for Peace Sign (to differentiate from RockNRoll)
    angle_pinky_pip = calculate_angle(landmarks[PINKY_MCP], landmarks[PINKY_PIP], landmarks[PINKY_DIP])
    pinky_flexed = False
    if angle_pinky_pip < 90: # Strongly flexed
        pinky_flexed = True
    elif angle_pinky_pip < 130: # Moderately flexed, check y-pos loosely
        pinky_flexed = landmarks[PINKY_TIP].y > landmarks[PINKY_MCP].y * 0.98

    # 5. Thumb Logic: Must be neutral/flexed AND not an aggressive thumbs-up pose
    angle_thumb_mcp = calculate_angle(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_IP])
    angle_thumb_ip = calculate_angle(landmarks[THUMB_MCP], landmarks[THUMB_IP], landmarks[THUMB_TIP])

    thumb_flexed_neutral = (angle_thumb_mcp < 150) or \
                           (landmarks[THUMB_TIP].y > landmarks[THUMB_MCP].y * 0.95) or \
                           (calculate_distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_MCP]) < calculate_distance(landmarks[THUMB_TIP], landmarks[THUMB_CMC])*0.8)

    is_thumb_angles_extended_for_aggressive_thumbs_up = angle_thumb_mcp > 140 and angle_thumb_ip > 140
    is_thumb_tip_high_vs_own_joints = landmarks[THUMB_TIP].y < landmarks[THUMB_IP].y * 0.95 and \
                                      landmarks[THUMB_TIP].y < landmarks[THUMB_MCP].y * 0.95
    is_thumb_tip_high_vs_wrist = landmarks[THUMB_TIP].y < landmarks[WRIST].y * 0.90
    avg_index_middle_mcp_y = (landmarks[INDEX_FINGER_MCP].y + landmarks[MIDDLE_FINGER_MCP].y) / 2.0
    is_thumb_tip_high_vs_relevant_mcps = landmarks[THUMB_TIP].y < avg_index_middle_mcp_y * 0.95

    is_aggressive_thumbs_up_thumb = is_thumb_angles_extended_for_aggressive_thumbs_up and \
                                   is_thumb_tip_high_vs_own_joints and \
                                   is_thumb_tip_high_vs_wrist and \
                                   is_thumb_tip_high_vs_relevant_mcps
    
    thumb_condition_met = thumb_flexed_neutral and (not is_aggressive_thumbs_up_thumb)

    if debug:
        print(f"PeaceSign V10 - Index Extended: {index_extended} (PIP: {angle_index_pip:.1f}, DIP: {angle_index_dip:.1f})")
        print(f"PeaceSign V10 - Middle Extended: {middle_extended} (PIP: {angle_middle_pip:.1f}, DIP: {angle_middle_dip:.1f})")
        print(f"PeaceSign V10 - Ring Flexed: {ring_flexed} (PIP: {angle_ring_pip:.1f}, TipY: {landmarks[RING_FINGER_TIP].y:.2f}, MCPY: {landmarks[RING_FINGER_MCP].y:.2f})")
        print(f"PeaceSign V10 - Pinky Flexed: {pinky_flexed} (PIP: {angle_pinky_pip:.1f}, TipY: {landmarks[PINKY_TIP].y:.2f}, MCPY: {landmarks[PINKY_MCP].y:.2f})")
        print(f"PeaceSign V10 - Thumb Flexed/Neutral (5a): {thumb_flexed_neutral} (MCP: {angle_thumb_mcp:.1f}, IP: {angle_thumb_ip:.1f})")
        print(f"PeaceSign V10 - Is Aggressive Thumbs Up Thumb (for 5b): {is_aggressive_thumbs_up_thumb}")
        print(f"PeaceSign V10 - Final Thumb Condition Met (5a AND not 5b): {thumb_condition_met}")

    return index_extended and middle_extended and ring_flexed and pinky_flexed and thumb_condition_met

def is_rock_n_roll_sign(landmarks, debug=False):
    # 1. Index Finger Extension
    angle_index_pip = calculate_angle(landmarks[INDEX_FINGER_MCP], landmarks[INDEX_FINGER_PIP], landmarks[INDEX_FINGER_DIP])
    angle_index_dip = calculate_angle(landmarks[INDEX_FINGER_PIP], landmarks[INDEX_FINGER_DIP], landmarks[INDEX_FINGER_TIP])
    index_extended = angle_index_pip > 150 and angle_index_dip > 150

    # 2. Pinky Finger Extension
    angle_pinky_pip = calculate_angle(landmarks[PINKY_MCP], landmarks[PINKY_PIP], landmarks[PINKY_DIP])
    angle_pinky_dip = calculate_angle(landmarks[PINKY_PIP], landmarks[PINKY_DIP], landmarks[PINKY_TIP])
    pinky_extended = angle_pinky_pip > 150 and angle_pinky_dip > 150

    # 3. Middle Finger Flexion
    angle_middle_pip = calculate_angle(landmarks[MIDDLE_FINGER_MCP], landmarks[MIDDLE_FINGER_PIP], landmarks[MIDDLE_FINGER_DIP])
    middle_flexed = angle_middle_pip < 100 and landmarks[MIDDLE_FINGER_TIP].y > landmarks[MIDDLE_FINGER_MCP].y * 0.95

    # 4. Ring Finger Flexion
    angle_ring_pip = calculate_angle(landmarks[RING_FINGER_MCP], landmarks[RING_FINGER_PIP], landmarks[RING_FINGER_DIP])
    ring_flexed = angle_ring_pip < 100 and landmarks[RING_FINGER_TIP].y > landmarks[RING_FINGER_MCP].y * 0.95

    # 5. Thumb Logic: Must be neutral/flexed AND not an aggressive thumbs-up pose
    angle_thumb_mcp = calculate_angle(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_IP])
    angle_thumb_ip = calculate_angle(landmarks[THUMB_MCP], landmarks[THUMB_IP], landmarks[THUMB_TIP])

    thumb_flexed_neutral = (angle_thumb_mcp < 150 and angle_thumb_ip < 175) or \
                           (landmarks[THUMB_TIP].y > landmarks[THUMB_MCP].y * 0.95) or \
                           (calculate_distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_MCP]) < calculate_distance(landmarks[THUMB_TIP], landmarks[THUMB_CMC])*0.9)

    is_thumb_angles_extended_for_aggressive_thumbs_up = angle_thumb_mcp > 140 and angle_thumb_ip > 140
    is_thumb_tip_high_vs_own_joints = landmarks[THUMB_TIP].y < landmarks[THUMB_IP].y * 0.95 and \
                                      landmarks[THUMB_TIP].y < landmarks[THUMB_MCP].y * 0.95
    is_thumb_tip_high_vs_wrist = landmarks[THUMB_TIP].y < landmarks[WRIST].y * 0.90
    avg_index_pinky_mcp_y = (landmarks[INDEX_FINGER_MCP].y + landmarks[PINKY_MCP].y) / 2.0
    is_thumb_tip_high_vs_relevant_mcps = landmarks[THUMB_TIP].y < avg_index_pinky_mcp_y * 0.95

    is_aggressive_thumbs_up_thumb = is_thumb_angles_extended_for_aggressive_thumbs_up and \
                                   is_thumb_tip_high_vs_own_joints and \
                                   is_thumb_tip_high_vs_wrist and \
                                   is_thumb_tip_high_vs_relevant_mcps
    
    thumb_condition_met = thumb_flexed_neutral and (not is_aggressive_thumbs_up_thumb)

    if debug:
        print(f"RockNRoll V10-R1 - Index Extended: {index_extended} (PIP: {angle_index_pip:.1f}, DIP: {angle_index_dip:.1f})")
        print(f"RockNRoll V10-R1 - Pinky Extended: {pinky_extended} (PIP: {angle_pinky_pip:.1f}, DIP: {angle_pinky_dip:.1f})")
        print(f"RockNRoll V10-R1 - Middle Flexed: {middle_flexed} (PIP: {angle_middle_pip:.1f}, TipY: {landmarks[MIDDLE_FINGER_TIP].y:.2f}, MCPY: {landmarks[MIDDLE_FINGER_MCP].y:.2f})")
        print(f"RockNRoll V10-R1 - Ring Flexed: {ring_flexed} (PIP: {angle_ring_pip:.1f}, TipY: {landmarks[RING_FINGER_TIP].y:.2f}, MCPY: {landmarks[RING_FINGER_MCP].y:.2f})")
        print(f"RockNRoll V10-R1 - Thumb Flexed/Neutral (5a): {thumb_flexed_neutral} (MCP: {angle_thumb_mcp:.1f}, IP: {angle_thumb_ip:.1f})")
        print(f"RockNRoll V10-R1 - Is Aggressive Thumbs Up Thumb (for 5b): {is_aggressive_thumbs_up_thumb}")
        print(f"RockNRoll V10-R1 - Final Thumb Condition Met (5a AND not 5b): {thumb_condition_met}")

    return index_extended and pinky_extended and middle_flexed and ring_flexed and thumb_condition_met

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Starting video stream (multi_gesture_detection_v10.py). Press ESC to exit.")
print("Debug messages for gesture detection will be printed to the console.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can_t receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture_text = "No Gesture"
    text_color = (0, 0, 255) # Red

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_thumbs_up(hand_landmarks.landmark, debug=True):
                gesture_text = "Thumbs Up!"
                text_color = (0, 255, 0) # Green
            elif is_rock_n_roll_sign(hand_landmarks.landmark, debug=True):
                gesture_text = "Rock N Roll!"
                text_color = (255, 105, 180) # Hot Pink
            elif is_peace_sign(hand_landmarks.landmark, debug=True):
                gesture_text = "Peace Sign!"
                text_color = (0, 255, 255) # Yellow
            
    
    cv2.putText(frame, gesture_text, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
    cv2.imshow("Multi-Gesture Detection v10", frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
print("Script finished.")
