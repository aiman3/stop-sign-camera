from collections import defaultdict
from src.util import draw_border, draw_result, setup_temp_dir, detect_violation, get_detection_area, ROOT_DIR, TEMP_DIR
import cv2
from configparser import ConfigParser
import os
from ultralytics import YOLO
import numpy as np
import time

GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

if __name__ == '__main__':
    # Read config file
    config = ConfigParser()
    if not os.path.exists(f'{ROOT_DIR}/config.ini'):
        raise FileNotFoundError(
            'Please rename ini file to "config.ini" before start')
    config.read(f'{ROOT_DIR}/config.ini')
    stream_url = config.get('rtsp', 'url')
    try:
        trigger_line = int(config.get('lines', 'trigger_line'))
        stop_line = int(config.get('lines', 'stop_line'))
    except ValueError:
        raise ValueError('Please set the trigger line and stop line')

    # Setup temp directory
    setup_temp_dir()

    # Setup YOLO model
    model = YOLO(f'{ROOT_DIR}/model/car.torchscript', task='detect')

    # Setup defaultdicts for tracking
    track_history = defaultdict(lambda: ([], []))
    trigger_frame = defaultdict(lambda: -1)
    stop_line_frame = defaultdict(lambda: -1)

    # Show realtime stream
    cap = cv2.VideoCapture(stream_url)
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Read detection area config
    x1, y1, x2, y2 = get_detection_area(frame_height, frame_width, config)

    while cap.isOpened():
        success, frame = cap.read()
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if success:
            # Detect traffic
            detection_frame = frame[y1:y2, x1:x2]
            results = model.track(detection_frame, persist=True)
            boxes = results[0].boxes.xywh.cuda()
            track_ids = results[0].boxes.id.int().cuda(
            ).tolist() if results[0].boxes.id != None else []

            frame = draw_result(frame, 'Detection Area',
                                (x1, y1), (x2, y2), BLUE, 10)

            # Draw trigger line
            frame = draw_result(
                frame, 'Trigger Line', (x1, trigger_line), (x2, trigger_line), GREEN, 3, 2)

            # Draw stop line
            frame = draw_result(
                frame, 'Stop Line', (x1, stop_line), (x2, stop_line), RED, 3, 2)

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                x, y = x+x1, y+y1
                frame_nums, track = track_history[track_id]
                track.append((float(x), float(y)))
                frame_nums.append(current_frame_num)
                if len(track) > 5*framerate:  # retain tracks for 5 seconds
                    track.pop(0)
                    frame_nums.pop(0)

                if trigger_frame[track_id] < 0 and trigger_line <= y < stop_line:
                    trigger_frame[track_id] = current_frame_num
                elif stop_line_frame[track_id] < 0 and y >= stop_line:
                    stop_line_frame[track_id] = current_frame_num

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                c = GRAY
                label = f'id:{track_id}'
                if trigger_frame[track_id] > -1:
                    c = GREEN
                    label = f'id:{track_id} {(current_frame_num-trigger_frame[track_id])/framerate:.1f} secs'
                if (stop_line_frame[track_id] > -1
                        and stop_line_frame[track_id] - trigger_frame[track_id] < 3*framerate):
                    c = RED
                    label = f'id:{track_id} {(stop_line_frame[track_id]-trigger_frame[track_id])/framerate:.1f} secs'
                cv2.polylines(frame, [points],
                              isClosed=False, color=c, thickness=5)

                # Draw bounding boxes
                frame = draw_result(frame, label, (int(
                    x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color=c, thickness=10)

            preview_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("Stop Sign Camera", preview_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
