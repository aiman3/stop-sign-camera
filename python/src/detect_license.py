import cv2
from ultralytics import YOLO
from util import ROOT_DIR, setup_dir, read_license_plate, draw_result, realesrgan
import easyocr
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('video_path', type=str)
args = parser.parse_args()

# Load the YOLOv8 model
model = YOLO(f'{ROOT_DIR}/model/license.torchscript', task='detect')

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
        results = model.predict(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # annotated_frame = frame

        license_plate_boxes = results[0].boxes.data.cpu().numpy()

        for i, box in enumerate(license_plate_boxes):
            x1, y1, x2, y2, conf, cls = box
            license_plate = frame[int(y1):int(y2), int(x1):int(x2)]

            #   Real-ESRGAN part, don't use, impacts performance
            #
            # input_image_path = f'{ROOT_DIR}/tmp_image/tmp.png'
            # cv2.imwrite(input_image_path, license_plate)
            # output_image_path = realesrgan(input_image_path)
            # plate_img = cv2.imread(output_image_path)
            # license_plate = plate_img

            p1, p2, plate_num, _ = read_license_plate(reader, license_plate)
            if plate_num != None:
                (plate_local_x1, plate_local_y1), \
                    (plate_local_x2, plate_local_y2) = p1, p2  # type: ignore
                plate_x1, plate_y1, plate_x2, plate_y2 = \
                    plate_local_x1 + x1, plate_local_y1 + y1, \
                    plate_local_x2 + x2, plate_local_y2 + y2
                annotated_frame = draw_result(
                    frame, plate_num, (plate_x1, plate_y1), (plate_x2, plate_y2))

            # print(f"License Plate {i+1} Text: {plate_num}")

        # annotated_frame = cv2.resize(annotated_frame, (1280, 720))
        annotated_frame = cv2.resize(annotated_frame, (1920, 1080))

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
