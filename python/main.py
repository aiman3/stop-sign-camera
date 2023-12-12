import argparse
from collections import defaultdict
from src.util import (Color, detect_license_plate, draw_result, get_capture_at_stop_line, get_direction, get_hline_or_vline,
                      get_trigger_lines, has_crossed_stop, has_crossed_trigger, read_license_plate, real_esrgan, resize_frame,
                      setup_dir, setup_temp_dir, track_vehicle,
                      get_detection_area, ROOT_DIR, TEMP_DIR)
import cv2
from configparser import ConfigParser, NoOptionError
import os
from ultralytics import YOLO
import numpy as np
import datetime
import easyocr
import argparse
import time


if __name__ == '__main__':
    # Get config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', nargs='?')
    args = parser.parse_args()
    config_path = args.config_path
    if config_path is None:
        config_path = f'{ROOT_DIR}/config.ini'

    # Get video feed from config file
    config = ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            'Please rename ini file to "config.ini" before start')
    config.read(config_path)
    video_path = config.get('video', 'source')
    try:
        output_path = config.get('video', 'output')
    except NoOptionError:
        output_path = ''

    # Show realtime stream
    cap = cv2.VideoCapture(video_path)
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output = None
    if output_path != '':
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(  # type: ignore
            *'XVID'), framerate, (1280, 720))

    # Read detection area and trigger lines from config file
    direction = get_direction(config)
    x1, y1, x2, y2 = get_detection_area(config, frame_height, frame_width)
    trigger_line, stop_line = get_trigger_lines(
        config, direction, x1, y1, x2, y2)
    capture_at_stop_line = get_capture_at_stop_line(config)

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

    # Setup some more useful stuff
    threshold = 5  # threshold for stop detection
    snapshop_hold = cv2.typing.MatLike  # type: ignore
    plate_num = None
    lp_x1, lp_y1, lp_x2, lp_y2 = None, None, None, None
    lp_frame = None
    lp_box = None
    downscale_ratio = frame_height // 720

    # Setups for preview
    ui_scale = frame_height // 720
    lp_display_w = 250 * ui_scale
    lp_display_h = 60 * ui_scale
    lp_display_x = x1-(lp_display_w+10)
    if lp_display_x < 0:
        lp_display_x = x2+10
        if lp_display_x > frame_width:
            lp_display_x = x2-(lp_display_w+10)
    lp_display_y = y1+(y2-y1)//2
    if lp_display_y < 0:
        lp_display_y = 0

    while cap.isOpened():
        start = time.time()
        success, frame = cap.read()
        if success:
            timestamp = datetime.datetime.now()
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Detect traffic
            detection_frame = frame[y1:y2, x1:x2]
            boxes, track_ids = track_vehicle(
                model_vehicle, detection_frame, (x1, y1), downscale_ratio, max_det=30)
            preview_frame = frame.copy()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 1*framerate:  # retain tracks for 1 seconds
                    track.pop(0)

                # Violation Detection
                if not is_triggered[track_id] and has_crossed_trigger(direction, trigger_line, stop_line, x, y):
                    is_triggered[track_id] = True
                    snapshop_hold = frame[int(
                        y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
                elif not is_crossed[track_id] and has_crossed_stop(direction, stop_line, x, y) and is_triggered[track_id]:
                    is_crossed[track_id] = True
                    if stationary_frame[track_id] < 3*framerate:
                        if capture_at_stop_line:
                            snapshop_hold = frame[int(
                                y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
                        cv2.imwrite(
                            f'{TEMP_DIR}/{timestamp.strftime("%Y-%m-%d")}/id_{track_id}_{timestamp.strftime("%Y%m%d-%H%M%S")}.png', snapshop_hold)
                        lp_box = detect_license_plate(
                            model_license, snapshop_hold)
                        if lp_box != None:
                            lp_x1, lp_y1, lp_x2, lp_y2 = lp_box
                            lp_frame = snapshop_hold[int(lp_y1):int(
                                lp_y2), int(lp_x1):int(lp_x2)]
                            cv2.imwrite(f'{TEMP_DIR}/tmp_lp.png', lp_frame)
                        else:
                            cv2.imwrite(f'{TEMP_DIR}/tmp_car.png',
                                        snapshop_hold)

                # TODO: Plate Number Recognition
                # cv2.imwrite(f'{TEMP_DIR}/tmp.png', lp_frame)
                # out_path = realesrgan(f'{TEMP_DIR}/tmp.png')
                # lp_frame = cv2.imread(out_path)
                # _, _, plate_num, _ = read_license_plate(
                # reader, lp_frame)

                # if plate_num != None:
                #     with open(f'{ROOT_DIR}/lp_nums.txt', 'a') as file:
                #         file.write(plate_num+'\n')
                #     preview_frame = draw_result(
                #         preview_frame, plate_num, (lp_x1, lp_y1), (lp_x2, lp_y2), Color.RED)

                # Check stationary
                if is_triggered[track_id] and len(track) > 2:
                    last_x, last_y = track[-2]
                    if not is_crossed[track_id] and abs(last_x - float(x)) + abs(last_y - float(y)) < threshold:
                        stationary_frame[track_id] += 1

                c = Color.GRAY
                label = f'id:{track_id}'
                #   crossed the trigger line
                if is_triggered[track_id]:
                    c = Color.GREEN
                #   crossed the stop line
                if is_crossed[track_id]:
                    c = Color.RED if stationary_frame[track_id] < 3 * \
                        framerate else Color.GREEN
                label = f'id:{track_id} {(stationary_frame[track_id])/framerate:.1f} secs'

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview_frame, [
                    points], isClosed=False, color=c, thickness=2*ui_scale)  # type: ignore

                # Draw bounding boxes
                preview_frame = draw_result(preview_frame, label, (int(
                    x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color=c, thickness=2*ui_scale)

            # Draw detection zone
            preview_frame = draw_result(preview_frame, 'Detection Area',
                                        (x1, y1), (x2, y2), Color.BLUE, 2*ui_scale, 2*ui_scale)

            # Draw trigger line
            line_p1, line_p2 = get_hline_or_vline(
                direction, trigger_line, (x1, y1), (x2, y2))
            preview_frame = draw_result(
                preview_frame, 'Trigger Line', line_p1, line_p2, Color.GREEN, ui_scale, ui_scale, text_x_offset=-370, text_y_offest=20)

            # Draw stop line
            line_p1, line_p2 = get_hline_or_vline(
                direction, stop_line, (x1, y1), (x2, y2))
            preview_frame = draw_result(
                preview_frame, 'Stop Line', line_p1, line_p2, Color.RED, ui_scale, ui_scale, text_x_offset=-300, text_y_offest=20)

            # Draw violator's plate (if detected) or violator's snapshot
            if lp_box != None and os.path.exists(f'{TEMP_DIR}/tmp_lp.png'):
                violator_frame = cv2.imread(f'{TEMP_DIR}/tmp_lp.png')
            elif lp_box == None and os.path.exists(f'{TEMP_DIR}/tmp_car.png'):
                violator_frame = cv2.imread(f'{TEMP_DIR}/tmp_car.png')
            violator_frame = resize_frame(
                violator_frame, 4*ui_scale, lp_display_w, lp_display_w)
            violator_frame_h, violator_frame_w = violator_frame.shape[:2]
            preview_frame[lp_display_y:(
                lp_display_y+violator_frame_h), lp_display_x:(lp_display_x+violator_frame_w)] = violator_frame
            preview_frame = cv2.putText(preview_frame, 'Last Bad Guy:', (lp_display_x, lp_display_y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9*ui_scale, Color.RED, 2*ui_scale)

            preview_frame = cv2.resize(preview_frame, (1280, 720))
            cv2.imshow("Stop Sign Camera", preview_frame)
            if output != None:
                output.write(preview_frame)
            if cv2.waitKey(1) == ord('q'):
                break

            # Remove old cars
            if len(track_history) > 100:
                to_remove = min(track_history.keys())
                track_history.pop(to_remove)
                is_triggered.pop(to_remove)
                is_crossed.pop(to_remove)
                stationary_frame.pop(to_remove)
                # snapshop_hold.pop(0)
        if frame is None:
            break
        end = time.time()
        print(f'loop time={(end-start)* 1000} ms')

    # Clean up
    cv2.destroyAllWindows()
    cap.release()
    if output != None:
        output.release()
