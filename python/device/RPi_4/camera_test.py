from picamera2 import Picamera2, Preview
import time

camera = Picamera2()
camera.start_preview(Preview.QT)
preview_config = camera.create_preview_configuration()
camera.configure(preview_config)
camera.start()
time.sleep(5)