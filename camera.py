import os
import cv2
import yaml
import numpy as np
from typing import List, Tuple, Dict


EXTENSION = ".png"
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_files")

class Camera:
    def __init__(self, name: str):
        self.name = name
        self.width: int = 0
        self.height: int = 0