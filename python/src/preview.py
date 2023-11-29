# Previews a image from dataset and draws a bounding box
# Usage: python preview.py [car_name] [-h]
# This script assumes following structure:
#   stop-sign-camera/
#   ├── dataset/
#   │   └── license_plate/
#   │       ├── annotations/
#   │       │   ├── car_name.xml
#   │       │   └── ...
#   │       └── images/
#   |           ├── car_name.png
#   │           └── ...
#   └── src/
#       └── preview.py

import xml.etree.ElementTree as ET
import os
from pathlib import Path
import cv2
import sys
import argparse
sys.path.append(str(Path(__file__).parent.parent))
from src.util import ROOT_DIR  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('car_name', default='Cars0', nargs='?', type=str)
args = parser.parse_args()

car_name = args.car_name
image_path = os.path.join(
    ROOT_DIR, 'dataset', 'license_plate', 'images', car_name + '.png')
annotation_path = os.path.join(
    ROOT_DIR, 'dataset', 'license_plate', 'annotations',  car_name + '.xml')

if os.path.exists(image_path):
    image = cv2.imread(image_path)
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    bbox = []
    for b in root.find('./object/bndbox'):
        bbox.append(int(b.text))

    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    cv2.imshow(car_name + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print('Error: "'+car_name+'.png" not found')
    print('This script assumes following structure:\n\
    stop-sign-camera/\n\
    ├── dataset/\n\
    │   └── license_plate/\n\
    │       ├── annotations/\n\
    │       │   ├── car_name.xml\n\
    │       │   └── ...\n\
    │       └── images/\n\
    |           ├── car_name.png\n\
    │           └── ...\n\
    └── src/\n\
        └── preview.py'
          )
    raise FileNotFoundError('"'+car_name+'.png" does not exist')
