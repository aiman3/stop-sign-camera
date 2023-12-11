from collections import defaultdict
import cv2
from ultralytics import YOLO
from util import ROOT_DIR, setup_dir
import argparse
import os
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str)
args = parser.parse_args()

# Load the YOLOv8 model
model = YOLO(f'{ROOT_DIR}/model/car.torchscript', task='detect')

# Open the video file
video_path = args.video_path
video_name = video_path.split(os.sep)[-1]
cap = cv2.VideoCapture(video_path)

# Setup output dir for cropped license plates
save_path = f'{Path(video_path).parent.absolute()}/detect/{video_name}/'
setup_dir(save_path)
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        ori_height, ori_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (1280, 720))
        results = model.track(frame_resized, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu(
        ).tolist() if results[0].boxes.id != None else []
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(
                230, 230, 230), thickness=5)

        annotated_frame = cv2.resize(annotated_frame, (1280, 720))

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
