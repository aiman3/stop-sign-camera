from pathlib import Path
import os


PROJECT_NAME = "stop-sign-camera"
ROOT_DIR = str([p for p in Path(__file__).parents
                if p.parts[-1] == PROJECT_NAME][0])
TEMP_DIR = os.path.join(ROOT_DIR, "tmp_image" + os.sep)


def setup_temp_dir():
    os.makedirs(os.path.dirname(TEMP_DIR), exist_ok=True)
