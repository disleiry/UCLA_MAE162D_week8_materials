import cv2
import numpy as np
import subprocess
import os

input_video = "example3_green.mp4"
temp_video    = "temp_result.avi"
output_video   = "result.mp4"

cap = cv2.VideoCapture(input_video)

# write AVI first (most stable with OpenCV)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_video, fourcc, 30.0, (640, 480))

# define green color range in HSV
lower_green = np.array([30, 150, 50])
upper_green = np.array([85, 255, 255])

print("[MAIN] Starting video processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # convert to HSV and create a mask for green objects
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()

print("[MAIN] Temporary AVI saved. Converting to MP4 using ffmpeg...")

# convert to H.264 MP4
subprocess.run(["ffmpeg", "-y", "-i", temp_video, "-vcodec", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# remove temporary AVI
os.remove(temp_video)

print("[MAIN] Video saved successfully!")
