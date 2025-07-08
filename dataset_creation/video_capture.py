import cv2
import time

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', 
                      cv2.VideoWriter_fourcc('M','J','P','G'),
                      60,
                      (frame_width, frame_height))

start_time = time.time()
duration = 60

print("Recording started...")

while int(time.time() - start_time) < duration:
    ret, frame = cap.read()
    if ret:
        out.write(frame)  
        cv2.imshow('Recording', frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped early by user.")
            break
    else:
        print("Failed to capture frame.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Recording finished and saved to output.avi")
