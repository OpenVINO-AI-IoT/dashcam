import sys
sys.path.append("../")
sys.path.append("/home/openalpr/openalpr/src/bindings/python")
from Service.LicensePlateIdentifier import LicensePlateIdentifier
from openalpr import Alpr
from Service.VideoCapture import VideoCapture
import argparse
import cv2
from time import sleep
import numpy as np

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
    args = parse_args()
    cap, out = get_video(args)
    alpr = Alpr("eu", "../ALPR/alpr_config/runtime_data/us.conf", "../ALPR/alpr_config/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)
    # while True:
    ret, frame = cap.read()
    if ret == True:
        key_pressed = cv2.waitKey(60)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, frame = cv2.threshold(frame,100,200,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # frame = np.dstack([frame]*3)
        lpr = LicensePlateIdentifier(frame, alpr)
        results = alpr.recognize_file(args.video)
        lpr.apply_alpr()
        if len(lpr.results['results']) > 0:
            results, candidates = lpr.extract_plates()
            for plate in results:
                print("Plate: ", plate)