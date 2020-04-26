"""Create the dataset
"""

import random
from glob import glob

# loading random images from the video samples
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw
from tqdm import tqdm

from utils import *

vids = glob("./vids/*")

# creating train
create_screenshots_from_video(video_path=vids[0], train=True)

# creating test
create_screenshots_from_video(video_path=vids[-1], train=False)