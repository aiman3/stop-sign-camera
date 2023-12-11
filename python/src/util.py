import math
from pathlib import Path
import os
import subprocess
import cv2
from matplotlib import pyplot as plt
import numpy as np
import re

PROJECT_NAME = "stop-sign-camera"
ROOT_DIR = str([p for p in Path(__file__).parents
                if p.parts[-1] == PROJECT_NAME][0])
TEMP_DIR = os.path.join(ROOT_DIR, "tmp_image" + os.sep)


def setup_temp_dir():
    os.makedirs(os.path.dirname(TEMP_DIR), exist_ok=True)


def setup_dir(dirname):
    os.makedirs(os.path.dirname(dirname), exist_ok=True)


def event_exit():
    cv2.destroyAllWindows()
    print('\nProgram exited gracefully, goodbye!')


def detect_violation(model, frame, offset):
    x1, y1 = offset
    # frame = cv2.resize(frame, (720, 720))
    results = model.track(frame, persist=True)
    calibrated_results = results  # TODO
    return calibrated_results


def get_detection_area(height, width, config) -> tuple[int, int, int, int]:
    # height, width = frame.shape[:2]
    det_x1 = int(config.get('frame', 'x1')) if config.get(
        'frame', 'x1') != '' else 0
    det_y1 = int(config.get('frame', 'y1')) if config.get(
        'frame', 'y1') != '' else 0
    det_x2 = int(config.get('frame', 'x2')) if config.get(
        'frame', 'x2') != '' else -1
    det_y2 = int(config.get('frame', 'y2')) if config.get(
        'frame', 'y2') != '' else -1
    if (det_x2, det_y2) == (-1, -1):
        print(
            f'Detection Area defined as: (0, 0), ({width}, {height})')
        return 0, 0, width, height
    if det_x1 > det_x2 or det_y1 > det_y2:
        print(
            f'Crop Area P1 cannot be greater than P2, resetting to full frame size')
        return 0, 0, width, height
    if not (0 < det_x1 <= width
            or 0 < det_y1 <= height
            or 0 < det_x2 <= width
            or 0 < det_y2 <= height):
        print(
            f'Crop Area out of frame, resetting to full frame size')
        return 0, 0, width, height
    print(
        f'Detection Area defined as: ({det_x1}, {det_y1}), ({det_x2}, {det_y2})')
    return det_x1, det_y1, det_x2, det_y2


def draw_line(im0):
    aligns = im0.shape
    p1 = (0, int((aligns[0]/2)))
    p2 = (aligns[1], int((aligns[0]/2)))
    circle = 2000
    source_img = cv2.circle(im0, p1, 10, (0, 255, 255), 6)
    source_img = cv2.circle(im0, p2, 10, (0, 255, 255), 6)
    # reference = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    fx = abs(p1[0] - p2[0])
    fy = abs(p1[1] - p2[1])

    if (p1[0]+p1[1]) < (p2[0]+p2[1]):
        if fx < fy:
            temp = abs(circle - p1[1]) / fy
            fx = (temp * fx) + p1[0]
            fy = (temp * fy) + p1[1]
            source_img = cv2.line(source_img, p1, p2, (0, 0, 255), 1)

        else:
            temp = abs(circle - p1[0]) / fx
            fx = (temp * fx) + p1[0]
            fy = (temp * fy) + p1[1]
            source_img = cv2.line(source_img, p1, p2, (0, 0, 255), 1)


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y),
             color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y),
             color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1),
             color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y),
             color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


# Preprocess cropped license plate image
def preprocess(img) -> cv2.typing.MatLike:
    img_lp = cv2.resize(img, (320, 320))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(
        img_gray_lp, 185, 255, cv2.THRESH_BINARY_INV)
    # img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    # img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    return img_binary_lp


def read_license_plate(reader, license_plate_crop) -> tuple[tuple[int, int], tuple[int, int], str, float] | tuple[None, None, None, None]:
    height, width = license_plate_crop.shape[:2]
    processed = preprocess(license_plate_crop)
    plate_num_results = reader.readtext(processed)
    for result in plate_num_results:
        bbox, plate_num, score = result
        (x1, y1), _, (x2, y2), _ = bbox
        x1, y1, x2, y2 = x1 * (width/320), y1 * \
            (height/320), x2 * (width/320), y2 * (height/320)
        plate_num = plate_num.upper()
        plate_num = re.sub('[^0-9a-zA-Z]+', '', plate_num)
        if len(plate_num) > 5:
            return (x1, y1), (x2, y2), plate_num, score
    return None, None, None, None


def draw_result(img, plate_num, p1, p2, color=(0, 255, 0), thickness=10, text_size=3) -> cv2.typing.MatLike:
    (x1, y1), (x2, y2) = p1, p2
    img = cv2.rectangle(img, (int(x1), int(y1)),
                        (int(x2), int(y2)), color, thickness)
    img = cv2.putText(img, plate_num, (int(x1), int(y1)-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 2, color, text_size)
    return img


async def realesrgan(input_image_path) -> str:
    output_folder = Path(input_image_path).parent.absolute()
    subprocess.run(["python",
                    f"{ROOT_DIR}/Real-ESRGAN/inference_realesrgan.py",
                    "-n",
                    "RealESRGAN_x4plus",
                    "-i",
                    input_image_path,
                    "-o",
                    output_folder,
                    "--outscale",
                    "3.5",
                    "--face_enhance"])
    filepath, ext = os.path.splitext(input_image_path)
    output_image_path = f'{filepath}_out{ext}'

    return output_image_path
