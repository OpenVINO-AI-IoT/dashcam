import cv2
import sys
sys.path.append("../../")
from utils import *

class VideoCapture(object):
    
    def __init__(self, video_source, video_output="person_detect.mp4"):
        self.video_source = video_source
        self.video_output = video_output
        self.cap = None
        self.out = None
        pass

    def create_capture(self):
        self.cap = cv2.VideoCapture(self.video_source)

    def create_output(self, fps=20.0, size=(200,200), source=None):
        # if is_linux():
        #     source = 0x0000021
        # elif is_windows() or is_macos():
        #     source = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter(self.video_output, source, fps, size, False)
