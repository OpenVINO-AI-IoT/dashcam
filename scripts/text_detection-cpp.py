import sys
import os
sys.path.append("../")
sys.path.append("/home/openalpr/openalpr/src/bindings/python")
import cv2
from time import sleep
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from Service.VideoCapture import VideoCapture

sys.path.append("../Text")
import libmain

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", type=str)
    parser.add_argument("--save_video", type=bool, default=False)
    
    return parser.parse_args()

def get_video(args):
    video_capture = VideoCapture(args.video)
    video_capture.create_capture()
    if args.save_video:
        video_capture.create_output()
    return video_capture.cap, video_capture.out

if __name__ == "__main__":
    # image = cv2.imread(sys.argv[1])

    args = parse_args()
    cap, out = get_video(args)
    while True:
        ret, frame = cap.read()
        if ret == True:
            key_pressed = cv2.waitKey(60)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, frame = cv2.threshold(frame,100,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # frame = np.dstack([frame]*3)
            txt = libmain.TextDetection(frame, 
            "/home/dashcam/Text/trained_classifierNM1.xml", 
            "/home/dashcam/Text/trained_classifierNM2.xml")
            txt.Run_Filters()
            image = txt.Get_Image()
            out.write(image)
            if key_pressed == 100:
                break
        else:
            break
    cap.release()
    out.release()