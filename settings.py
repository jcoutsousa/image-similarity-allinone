from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# ML Model config
#MODEL_DIR = ROOT / 'weight'
#DETECTION_MODEL = MODEL_DIR / 'best.pt'
#SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'


# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'eiffel_1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'Eiffel-Tower-Paris.jpg'
