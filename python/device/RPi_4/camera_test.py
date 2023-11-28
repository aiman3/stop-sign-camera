from picamera2 import Picamera2, Preview
from pathlib import Path
import time
import sys
import cv2
import os

PROJECT_NAME = "stop-sign-camera"
ROOT_DIR = [p for p in Path(__file__).parents
            if p.parts[-1] == PROJECT_NAME][0]

picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={
        "size": (1600, 1200),
        "format": "RGB888"
    })
picam2.configure(preview_config)
picam2.start()
time.sleep(5)
temp_folder_path = os.path.join(ROOT_DIR, "tmp_image" + os.sep)
os.makedirs(os.path.dirname(temp_folder_path), exist_ok=True)
picam2.capture_file(temp_folder_path + "test.png")
while True:
    try:
        frame = picam2.capture_array()
        cv2.imshow("picamera2", frame)
    except (KeyboardInterrupt, SystemExit):
        picam2.capture_file(temp_folder_path + "exit.png")
        cv2.destroyAllWindows()
        print('\nProgram Stopped Manually!')
        raise
