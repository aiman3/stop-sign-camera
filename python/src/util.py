from configparser import NoOptionError
from enum import Enum
from pathlib import Path
import os
import subprocess
import cv2
import re
import glob

PROJECT_NAME = "stop-sign-camera"
ROOT_DIR = str([p for p in Path(__file__).parents
                if p.parts[-1] == PROJECT_NAME][0])
TEMP_DIR = os.path.join(ROOT_DIR, "tmp_image")


class Color():
    GRAY = (180, 180, 180)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)


class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


direction_dict = {'n': Direction.NORTH,
                  's': Direction.SOUTH,
                  'e': Direction.EAST,
                  'w': Direction.WEST
                  }


def setup_temp_dir():
    os.makedirs(os.path.dirname(TEMP_DIR), exist_ok=True)


def setup_dir(dirname):
    os.makedirs(os.path.dirname(dirname), exist_ok=True)


def event_exit():
    cv2.destroyAllWindows()
    print('\nProgram exited gracefully, goodbye!')


def track_vehicle(model, frame, offset, downsize_ratio=4, max_det=4):
    x1, y1 = offset
    height, width = frame.shape[:2]
    frame = cv2.resize(
        frame, (int(width/downsize_ratio), int(height/downsize_ratio)))
    results = model.track(frame, persist=True, max_det=max_det)
    boxes = results[0].boxes.xywh.cuda()
    track_ids = results[0].boxes.id.int().cuda(
    ).tolist() if results[0].boxes.id != None else []
    for box in boxes:
        box[0], box[1] = box[0]*downsize_ratio+x1, box[1]*downsize_ratio+y1
        box[2], box[3] = box[2]*downsize_ratio, box[3]*downsize_ratio
    return boxes, track_ids


def detect_license_plate(model, frame, offset=0):
    results = model.predict(frame)
    boxes = results[0].boxes.xyxy.cuda()
    if len(boxes) <= 0:
        return None
    x1, y1, x2, y2 = boxes[0]
    return x1, y1, x2, y2


def get_detection_area(config, height, width) -> tuple[int, int, int, int]:
    try:
        det_x1 = int(config.get('frame', 'x1'))
        det_y1 = int(config.get('frame', 'y1'))
        det_x2 = int(config.get('frame', 'x2'))
        det_y2 = int(config.get('frame', 'y2'))
    except ValueError:
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


def get_trigger_lines(config, direction, det_x1, det_y1, det_x2, det_y2) -> tuple[int, int]:
    try:
        trigger_line = int(config.get('lines', 'trigger_line'))
        stop_line = int(config.get('lines', 'stop_line'))
    except ValueError:
        raise ValueError('Please set the trigger line and stop line')
    if direction is Direction.NORTH or direction is Direction.SOUTH:
        if not det_y1 <= trigger_line < det_y2:
            raise ValueError('trigger_line not within detection box')
        if not det_y1 <= stop_line < det_y2:
            raise ValueError('stop_line not within detection box')
    elif direction is Direction.EAST or direction is Direction.WEST:
        if not det_x1 <= trigger_line < det_x2:
            raise ValueError('trigger_line not within detection box')
        if not det_x1 <= stop_line < det_x2:
            raise ValueError('stop_line not within detection box')
    else:
        raise RuntimeError(f'unknown direction {direction}')
    return trigger_line, stop_line


def get_snapshot_line(config, direction, trigger_line, stop_line) -> int:
    try:
        snapshot_line = int(config.get('lines', 'snapshot_line'))
    except (ValueError, NoOptionError):
        snapshot_line = (trigger_line + stop_line) // 2
    if direction not in Direction:
        raise RuntimeError(f'unknown direction {direction}')
    if (
        (direction is Direction.NORTH and not stop_line <= snapshot_line <= trigger_line) or
        (direction is Direction.SOUTH and not trigger_line <= snapshot_line <= stop_line) or
        (direction is Direction.EAST and not stop_line <= snapshot_line <= trigger_line) or
        (direction is Direction.WEST and not trigger_line <= snapshot_line <= stop_line)
    ):
        print(
            f'snapshot_line out of range, resetting to trigger_line: {trigger_line}')
        snapshot_line = (trigger_line + stop_line) // 2

    return snapshot_line


def get_direction(config):
    direction = config.get('lines', 'direction')
    direction = direction.lower()
    if direction not in direction_dict:
        raise ValueError(
            '[lines] direction= should be either "n","s","e", or "w" (case insensitive)')
    return direction_dict[direction]


def get_hline_or_vline(direction, line, det_p1, det_p2):
    x1, y1 = det_p1
    x2, y2 = det_p2
    if direction is Direction.NORTH or direction is Direction.SOUTH:
        return (x1, line), (x2, line)
    elif direction is Direction.EAST or direction is Direction.WEST:
        return (line, y1), (line, y2)
    else:
        raise RuntimeError(f'unknown direction {direction}')


def has_crossed_trigger(direction: Direction, trigger: int, stop: int, x: int, y: int) -> bool:
    if direction is Direction.NORTH:
        return stop <= y < trigger
    elif direction is Direction.SOUTH:
        return trigger <= y < stop
    elif direction is Direction.EAST:
        return trigger <= x < stop
    elif direction is Direction.WEST:
        return stop <= x < trigger
    else:
        raise RuntimeError(f'unknown direction {direction}')


def has_crossed_line(direction: Direction, stop: int, x: int, y: int) -> bool:
    if direction is Direction.NORTH:
        return y <= stop
    elif direction is Direction.SOUTH:
        return y >= stop
    elif direction is Direction.EAST:
        return x >= stop
    elif direction is Direction.WEST:
        return x <= stop
    else:
        raise RuntimeError(f'unknown direction {direction}')


def remove_temp_image(date, id):
    paths = glob.glob(f'{TEMP_DIR}/{date}/id_{id}_*')
    for path in paths:
        Path(path).unlink(missing_ok=True)


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


def resize_frame(img, scale: float, max_w: int = 720, max_h: int = 720) -> cv2.typing.MatLike:  # type: ignore
    img_h, img_w = img.shape[:2]
    wh_ratio = img_w/img_h
    resize_w, resize_h = img_w*scale, img_h*scale
    if resize_w > max_w or resize_h > max_h:
        if resize_w > resize_h:
            resize_w = max_w
            resize_h = max_w / wh_ratio
        else:
            resize_w = max_h * wh_ratio
            resize_h = max_h
    return cv2.resize(img, (int(resize_w), int(resize_h)))


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
def preprocess(img, w, h) -> cv2.typing.MatLike:  # type: ignore
    img_lp = cv2.resize(img, (w*10, h*10))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(
        img_gray_lp, 200, 255, cv2.THRESH_BINARY_INV)
    # img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    # img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    return img_binary_lp


def read_license_plate(reader, license_plate_crop) -> tuple[tuple[int, int], tuple[int, int], str, float] | tuple[None, None, None, None]:
    height, width = license_plate_crop.shape[:2]
    processed = preprocess(license_plate_crop, width, height)
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


def draw_result(img, plate_num, p1, p2, color=(0, 255, 0), thickness=10, text_size=3.0, text_x_offset=0, text_y_offest=-10) -> cv2.typing.MatLike:  # type: ignore
    (x1, y1), (x2, y2) = p1, p2
    img = cv2.rectangle(img, (int(x1), int(y1)),
                        (int(x2), int(y2)), color, thickness)
    img = cv2.putText(img, plate_num, (int(x1)+text_x_offset, int(y1)+text_y_offest),
                      cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness)
    return img


def real_esrgan(input_image_path) -> str:
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
