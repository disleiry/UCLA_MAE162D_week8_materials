# import useful libraries
import cv2
import subprocess
import os
from yolo_utils import *
from picamera2 import Picamera2

# video file names
temp_video = "temp_recording.avi"
output_video = "recording.mp4"

# check OpenCV + CUDA
print("OpenCV version :", cv2.__version__)
print("Available CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount(), "\n")

# load class names
obj_file = './obj.names'
classNames = read_classes(obj_file)
print("Classes' names :", classNames, "\n")

# load YOLO model
modelConfig_path = './cfg/yolov4.cfg'
modelWeights_path = './weights/yolov4.weights'

neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)
neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

confidenceThreshold = 0.5
nmsThreshold = 0.1

network = neural_net
height, width = 128, 128

# ---------- Traffic light + color detection settings ----------
TRAFFIC_LIGHT_CLASS_ID = 0  # change if needed based on obj.names

LOWER_RED1 = (0, 120, 70)
UPPER_RED1 = (10, 255, 255)
LOWER_RED2 = (170, 120, 70)
UPPER_RED2 = (180, 255, 255)

MIN_RED_CONTOUR_AREA = 150

# initialize Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
picam2.start()

# setup Video Writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_video, fourcc, 10.0, (640, 480))

print("[MAIN] Recording started... Press Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # object detection
        outputs = convert_to_blob(frame, network, height, width)
        bounding_boxes, class_objects, confidence_probs = object_detection(
            outputs, frame, confidenceThreshold)

        # -------- Traffic light color detection --------
        for i in range(len(bounding_boxes)):

            print(f"[Debug] Detected: Class={class_objects[i]}, Confidence={confidence_probs[i]:.2f}")

            # 1) Filter detections
            if class_objects[i] != TRAFFIC_LIGHT_CLASS_ID:
                continue

            # 2) Crop bounding box (FIXED: force integer indices)
            x, y, w, h = bounding_boxes[i]

            x1 = int(max(0, round(x)))
            y1 = int(max(0, round(y)))
            x2 = int(min(frame.shape[1], round(x + w)))
            y2 = int(min(frame.shape[0], round(y + h)))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]

            # 3) Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 4) Create red mask
            mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
            mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # clean noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

            # 5) Check contour size
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            red_detected = False
            for c in contours:
                if cv2.contourArea(c) >= MIN_RED_CONTOUR_AREA:
                    red_detected = True
                    break

            # 6) Determine status
            if red_detected:
                print("[STATUS] Red light detected!")
                cv2.putText(frame, "RED", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # -------- Draw YOLO boxes --------
        indices = nms_bbox(
            bounding_boxes,
            confidence_probs,
            confidenceThreshold,
            nmsThreshold
        )

        box_drawing(
            frame,
            indices,
            bounding_boxes,
            class_objects,
            confidence_probs,
            classNames,
            color=(0, 255, 255),
            thickness=2
        )

        # write frame
        out.write(frame)

except KeyboardInterrupt:
    print("\n[MAIN] Stopping recording...")

# cleanup
out.release()
picam2.close()

print("[MAIN] Converting to MP4 using ffmpeg...")

subprocess.run([
    "ffmpeg", "-y", "-i", temp_video,
    "-vcodec", "libx264",
    "-preset", "ultrafast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    output_video
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

os.remove(temp_video)

print("[MAIN] Video saved successfully as", output_video)
