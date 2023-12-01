# Previews a image from dataset and draws a bounding box
# Usage: python preview.py [dataset] [filename] [-h]
# This script assumes following structure:
#   stop-sign-camera/
#   ├── dataset_raw/
#   │   └── <dataset>/
#   │       ├── annotations/
#   │       │   ├── <filename>.xml
#   │       │   └── ...
#   │       └── images/
#   |           ├── <filename>.png
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
parser.add_argument('dataset', default='license_plate', nargs='?', type=str)
parser.add_argument('filename', default='Cars0', nargs='?', type=str)
args = parser.parse_args()

filename = args.filename
dataset = args.dataset
image_path = os.path.join(
    ROOT_DIR, 'dataset_raw', dataset, 'images', filename + '.png')
annotation_path = os.path.join(
    ROOT_DIR, 'dataset_raw', dataset, 'annotations',  filename + '.xml')

if os.path.exists(image_path):
    image = cv2.imread(image_path)
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    bboxs = []
    for b in root.findall('./object/bndbox'):
        bboxs.append(b)
    for bbox in bboxs:
        points = []
        for b in bbox:
            points.append(int(b.text))
        x1, y1 = points[0], points[1]
        x2, y2 = points[2], points[3]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    cv2.imshow(filename + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print('Error: "'+filename+'.png" not found')
    print('This script assumes following structure:\n\
    stop-sign-camera/\n\
    ├── dataset_raw/\n\
    │   └── <dataset>/\n\
    │       ├── annotations/\n\
    │       │   ├── <filename>.xml\n\
    │       │   └── ...\n\
    │       └── images/\n\
    |           ├── <filename>.png\n\
    │           └── ...\n\
    └── src/\n\
        └── preview.py'
          )
    raise FileNotFoundError('"'+filename+'.png" does not exist')
