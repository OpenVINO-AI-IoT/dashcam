import time
import cv2
import sys
sys.path.append("../")
sys.path.append("/home/openalpr/openalpr/src/bindings/python")
from Service.LicensePlateIdentifier import LicensePlateIdentifier
# from openalpr import Alpr
from Service.VideoCapture import VideoCapture
import argparse

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

    cap, out = get_video(parse_args())
    # alpr = Alpr("br", "../ALPR/alpr_config/runtime_data/config/gb.conf", 
    # "../ALPR/alpr_config/runtime_data")
    # if not alpr.is_loaded():
    #     print("Error loading OpenALPR")
    #     sys.exit(1)

    # Cam properties
    if cap.isOpened() is not True:
        print("Cannot open camera. Exiting.")
        quit()

    # Loop it
    idx = 0
    while True:
        ret, frame = cap.read()
        if ret == True:
            key_pressed = cv2.waitKey(60)
            # lpr = LicensePlateIdentifier(frame, alpr)
            # lpr.apply_alpr()
            # print(lpr.results)
            cv2.imshow('frame', frame)
            # if len(lpr.results['results']) > 0:
            #     plates, confidences = lpr.extract_plates()
            #     print(plates, confidences, lpr.results['vehicle_region'])
            print(key_pressed)
            if key_pressed == 100:
                cv2.imwrite("image-"+idx.__str__()+".png", frame)
                idx += 1
        else:
            break

    cap.release()