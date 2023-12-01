import cv2
from ultralytics import YOLO
from util import ROOT_DIR, setup_dir
import easyocr
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str)
args = parser.parse_args()

# Load the YOLOv8 model
model = YOLO(f'{ROOT_DIR}/model/best.torchscript', task='detect')

# Open the video file
video_path = args.video_path
video_name = video_path.split(os.sep)[-1]
cap = cv2.VideoCapture(video_path)

# Setup reader
reader = easyocr.Reader(['en'])

# Setup output dir for cropped license plates
save_path = f'{Path(video_path).parent.absolute()}/detect/{video_name}/'
setup_dir(save_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, conf=0.15, max_det=4)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        license_plate_boxes = results[0].boxes.data.cpu().numpy()

        for i, box in enumerate(license_plate_boxes):
            x1, y1, x2, y2, conf, cls = box
            license_plate = frame[int(y1):int(y2), int(x1):int(x2)]

            plate_filename = f'{save_path}/frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}_license_plate_{i}.png'
            cv2.imwrite(plate_filename, license_plate)

            plate_num_results = reader.readtext(license_plate)
            # print(plate_num_results)
            # print(f"License Plate {i+1} Text: {plate_num_results[0][1]}")

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
