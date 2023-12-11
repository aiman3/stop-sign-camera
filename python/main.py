from collections import defaultdict
import math
from src.util import detect_license_plate, draw_border, draw_result, get_trigger_lines, read_license_plate, realesrgan, remove_temp_image, setup_dir, setup_temp_dir, track_vehicle, get_detection_area, ROOT_DIR, TEMP_DIR
import cv2
from configparser import ConfigParser
import os
from ultralytics import YOLO
import numpy as np
import datetime
import easyocr

GRAY = (180, 180, 180)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

if __name__ == '__main__':
    # Get video feed from config file
    config = ConfigParser()
    if not os.path.exists(f'{ROOT_DIR}/config.ini'):
        raise FileNotFoundError(
            'Please rename ini file to "config.ini" before start')
    config.read(f'{ROOT_DIR}/config.ini')
    video_path = config.get('video', 'source')
    output_path = config.get('video', 'output')

    # Show realtime stream
    cap = cv2.VideoCapture(video_path)
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output = None
    if output_path != '':
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
            *'XVID'), framerate, (1280, 720))

    # Read detection area and trigger lines from config file
    x1, y1, x2, y2 = get_detection_area(frame_height, frame_width, config)
    trigger_line, stop_line = get_trigger_lines(config, y1, y2)

    # Get today's date
    date = datetime.datetime.now()

    # Setup temp directories
    setup_temp_dir()
    setup_dir(f'{TEMP_DIR}/{date.strftime("%Y-%m-%d")}/')

    # Setup YOLO model, EasyOCR
    model_vehicle = YOLO(
        f'{ROOT_DIR}/model/car.torchscript', task='detect')
    model_license = YOLO(
        f'{ROOT_DIR}/model/license.torchscript', task='detect')
    reader = easyocr.Reader(['en'])

    # Setup defaultdicts for tracking
    track_history = defaultdict(lambda: [])
    is_triggered = defaultdict(lambda: False)
    is_crossed = defaultdict(lambda: False)
    stationary_frame = defaultdict(lambda: 0)

    # Setup stationary threshold
    threshold = 5
    snapshop_hold = cv2.typing.MatLike
    plate_num = None
    lp_x1, lp_y1, lp_x2, lp_y2 = None, None, None, None
    lp_frame = None

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            time = datetime.datetime.now()
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Detect traffic
            detection_frame = frame[y1:y2, x1:x2]
            boxes, track_ids = track_vehicle(
                model_vehicle, detection_frame, (x1, y1), 6, max_det=3)
            preview_frame = frame.copy()
            preview_frame = draw_result(preview_frame, 'Detection Area',
                                        (x1, y1), (x2, y2), BLUE, 10)

            # Draw trigger line
            preview_frame = draw_result(
                preview_frame, 'Trigger Line', (x1, trigger_line), (x2, trigger_line), GREEN, 3, 3, text_x_offset=-370, text_y_offest=20)

            # Draw stop line
            preview_frame = draw_result(
                preview_frame, 'Stop Line', (x1, stop_line), (x2, stop_line), RED, 3, 3, text_x_offset=-300, text_y_offest=20)

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 1*framerate:  # retain tracks for 1 seconds
                    track.pop(0)

                # Violation Detection
                if not is_triggered[track_id] and trigger_line <= y < stop_line:
                    is_triggered[track_id] = True
                    snapshop_hold = frame[int(
                        y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
                elif not is_crossed[track_id] and y >= stop_line:
                    is_crossed[track_id] = True
                    if stationary_frame[track_id] < 3*framerate:
                        cv2.imwrite(
                            f'{TEMP_DIR}/{time.strftime("%Y-%m-%d")}/id_{track_id}_{time.strftime("%Y%m%d-%H%M%S")}.png', snapshop_hold)
                        lp_x1, lp_y1, lp_x2, lp_y2 = detect_license_plate(
                            model_license, snapshop_hold)
                        lp_frame = snapshop_hold[int(lp_y1):int(
                            lp_y2), int(lp_x1):int(lp_x2)]
                        cv2.imwrite(f'{TEMP_DIR}/tmp.png', lp_frame)

                # cv2.imwrite(f'{TEMP_DIR}/tmp.png', lp_frame)
                # out_path = realesrgan(f'{TEMP_DIR}/tmp.png')
                # lp_frame = cv2.imread(out_path)
                # _, _, plate_num, _ = read_license_plate(
                # reader, lp_frame)

                if plate_num != None:
                    with open(f'{ROOT_DIR}/lp_nums.txt', 'a') as file:
                        file.write(plate_num+'\n')
                    preview_frame = draw_result(
                        preview_frame, plate_num, (lp_x1, lp_y1), (lp_x2, lp_y2), RED)

                if is_triggered[track_id] and len(track) > 2:
                    last_x, last_y = track[-2]
                    if not is_crossed[track_id] and abs(last_x - float(x)) + abs(last_y - float(y)) < threshold:
                        stationary_frame[track_id] += 1

                c = GRAY
                label = f'id:{track_id}'
                #   crossed the trigger line
                if is_triggered[track_id]:
                    c = GREEN
                #   crossed the stop line
                if is_crossed[track_id]:
                    c = RED if stationary_frame[track_id] < 3 * \
                        framerate else GREEN
                label = f'id:{track_id} {(stationary_frame[track_id])/framerate:.1f} secs'

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview_frame, [points],
                              isClosed=False, color=c, thickness=5)

                # Draw bounding boxes
                preview_frame = draw_result(preview_frame, label, (int(
                    x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color=c, thickness=10)

            if os.path.exists(f'{TEMP_DIR}/tmp.png'):
                lp_frame = cv2.imread(f'{TEMP_DIR}/tmp.png')
                lp_frame_resized = cv2.resize(
                    lp_frame, (lp_frame.shape[1]*6, lp_frame.shape[0]*6))
                preview_frame[y1+(y2-y1)//2:y1+(y2-y1)//2+lp_frame_resized.shape[0],
                              x1-10-lp_frame_resized.shape[1]:x1-10] = lp_frame_resized
                preview_frame = cv2.putText(preview_frame, 'Last Bad Guy:', (x1-10-lp_frame_resized.shape[1], y1+(y2-y1)//2-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 5)

            preview_frame = cv2.resize(preview_frame, (1280, 720))
            cv2.imshow("Stop Sign Camera", preview_frame)
            if output != None:
                output.write(preview_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            if len(track_history) > 100:
                track_history.pop(0)
                is_triggered.pop(0)
                is_crossed.pop(0)
                stationary_frame.pop(0)
        if frame is None:
            break

    # Clean up
    cv2.destroyAllWindows()
    cap.release()
    if output != None:
        output.release()
