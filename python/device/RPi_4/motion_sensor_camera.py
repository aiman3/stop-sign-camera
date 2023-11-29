# A script that capture an 16MP image (4656x3496px) when any motion is detected

from datetime import datetime
from gpiozero import MotionSensor
from picamera2 import Picamera2
from libcamera import controls
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.util import setup_temp_dir, TEMP_DIR  # noqa: E402


class MotionSensorCamera:
    def __init__(self, pir_pin=4) -> None:
        self.pir = MotionSensor(pir_pin)
        self.camera = Picamera2()
        self.preview_config = self.camera.create_preview_configuration(
            main={
                "size": (1920, 1080),
                "format": "RGB888"
            }
        )
        self.capture_config = self.camera.create_still_configuration()

    def start(self) -> None:
        self.camera.configure(self.preview_config)
        self.camera.start()
        self.camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})

    def capture(self) -> None:
        print('Motion Detected! Capturing the image')
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.camera.switch_mode_and_capture_file(
            self.capture_config,
            TEMP_DIR + 'motion_detected_' + current_time + '.png')
        self.camera.switch_mode(self.preview_config)
        print('Image saved at ' + TEMP_DIR +
              'motion_detected_' + current_time + '.png')


if __name__ == '__main__':
    msc = MotionSensorCamera()
    msc.start()
    msc.pir.when_motion = msc.capture
    while True:
        try:
            print('Detecting motion...')
            msc.pir.wait_for_motion(2)
            msc.pir.wait_for_no_motion(2)
        except (KeyboardInterrupt, SystemExit):
            print('Program exited gracefully, goodbye!')
            break
