from picamera2 import Picamera2
from libcamera import controls
from pathlib import Path
import cv2
import os
from src.util import setup_temp_dir, TEMP_DIR


def event_exit():
    picam2.capture_file(TEMP_DIR + "test_exit.png")
    cv2.destroyAllWindows()
    print('\nTest aborted, goodbye!')


# Main
setup_temp_dir()
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={
        "size": (1920, 1080),
        "format": "BGR888"
    },
    lores={
        "size": (1280, 720),
        "format": "YUV420"
    })
picam2.configure(preview_config)
picam2.start()
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.capture_file(TEMP_DIR + "test_start.png")
print('Camera Started Successfully!', 'Press q or Ctrl+C to abort')
while True:
    try:
        frame = picam2.capture_array("lores")
        cv2.imshow("picamera2", frame)
        if cv2.waitKey(1) == ord('q'):
            event_exit()
            break
    except (KeyboardInterrupt, SystemExit):
        event_exit()
        break
