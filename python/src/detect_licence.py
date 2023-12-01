import cv2
from ultralytics import YOLO
from util import ROOT_DIR
import easyocr

# Load the YOLOv8 model
model = YOLO(ROOT_DIR+'/model/best.torchscript', task='detect')

# Open the video file
video_path = ROOT_DIR+'/tmp_image/MNn9qKG2UFI.webm'
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, conf=0.15, max_det=4)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

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
