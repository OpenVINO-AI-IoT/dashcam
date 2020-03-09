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
    # image = cv2.imread(sys.argv[1])

    args = parse_args()
    cap, out, size = get_video(args)
    state = np.zeros((size[1],size[0],3))
    txt = libmain.TextDetection()
    idx = 0
    while True:
        ret, frame = cap.read()
        if ret == True:
            txt.Initialize(frame, 
            "/home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/License_Plate_Recognition/Dashcam-project/Text/trained_classifierNM1.xml", 
            "/home/aswin/Documents/Courses/Udacity/Intel-Edge/Work/EdgeApp/License_Plate_Recognition/Dashcam-project/Text/trained_classifierNM2.xml")
            txt.Run_Filters()
            image = txt.Groups_Draw(np.zeros_like(frame))
            out.write(image)
            idx += 1
            if idx % 100 == 0:
                print(idx)
        else:
            break
    cap.release()
    out.release()