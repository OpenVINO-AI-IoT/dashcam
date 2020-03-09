import sys
sys.path.append("../")
from Service.LicensePlateIdentifier import LicensePlateIdentifier
from Service.VideoCapture import VideoCapture
import argparse
import cv2
import os
from time import sleep, time
import numpy as np
from ALPR import libalpr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", type=str)
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--fps", type=float, default=False)
    parser.add_argument("--save_path", type=str, default=False)
    parser.add_argument("--is_color", type=bool, default=True)
    
    return parser.parse_args()

def get_video(args):
    video_capture = VideoCapture(args.video, video_output=args.save_path)
    video_capture.create_capture()
    width = int(video_capture.cap.get(3))
    height = int(video_capture.cap.get(4))
    size=(width,height)
    if args.save_video:
        video_capture.create_output(fps=args.fps, size=size, isColor=args.is_color, source=cv2.VideoWriter_fourcc(*"XVID"))
    return video_capture.cap, video_capture.out, size

if __name__ == "__main__":
    args = parse_args()
    cap, out, size = get_video(args)
    alpr = lib_alpr.ALPRImageDetect()
    alpr.Attributes(os.path.abspath("../ALPR/alpr_config/runtime_data/gb.conf"), 
    "eu", "", os.path.abspath("../ALPR/alpr_config/runtime_data"))
    while True:
        ret, frame = cap.read()
        if ret == True:
            alpr.SetFrame(frame)
            t1 = time()
            plates = alpr.LicensePlate_Matches(10, None, False) # (182,358,345,410)
            key_pressed = cv2.waitKey(60)
            placements = alpr.getPlacements()
            print(plates, placements)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, frame = cv2.threshold(frame,100,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # frame = np.dstack([frame]*3)
        else:
            break