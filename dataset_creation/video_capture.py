import cv2
import time

# Set up video capture
cap = cv2.VideoCapture(0)  # 0 = default webcam

# Get the video frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', 
                      cv2.VideoWriter_fourcc('M','J','P','G'),
                      60,  # frames per second
                      (frame_width, frame_height))

# Set timer
start_time = time.time()
duration = 60  # 60 seconds

print("Recording started...")

while int(time.time() - start_time) < duration:
    ret, frame = cap.read()
    if ret:
        out.write(frame)  # write frame to output file
        cv2.imshow('Recording', frame)  # show live preview

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped early by user.")
            break
    else:
        print("Failed to capture frame.")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Recording finished and saved to output.avi")
