# Usage: run this script on RPi first, then run "python ./python/src/connect_stream.py"
# to preview the stream
#   python stream.py

import socket
from libcamera import controls
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

picam2 = Picamera2()
video_config = picam2.create_video_configuration({"size": (3840, 2160)})
picam2.configure(video_config)
encoder = JpegEncoder()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 8554))
    sock.listen()

    picam2.encoders = encoder

    conn, addr = sock.accept()
    stream = conn.makefile("wb")
    encoder.output = FileOutput(stream)
    picam2.start_encoder(encoder)
    picam2.start()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    while True:
        try:
            pass
        except (KeyboardInterrupt, SystemExit):
            break
    picam2.stop()
    picam2.stop_encoder()
    conn.close()
