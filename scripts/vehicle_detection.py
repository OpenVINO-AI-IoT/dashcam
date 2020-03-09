import numpy as np
import cv2
import os
import sys
sys.path.append("../")
import NetworkService
from Vehicle.VehicleDetection import VehicleDetection
from Service.VideoCapture import VideoCapture
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", type=str)
    parser.add_argument("--save_video", type=bool, default=False)
    parser.add_argument("--fps", type=float, default=False)
    parser.add_argument("--save_path", type=str, default=False)
    parser.add_argument("--is_color", type=bool, default=True)
    parser.add_argument(
        '-m', '--model',
        help="the deep learning model",
        type=str,
        default="../Vehicle/model/model.xml",
        required=True
    )
    
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

def process_network(net, frame, input_shape):
    image = cv2.resize(frame, tuple(input_shape[0:2]))
    image = image.reshape(2,0,1)
    net.async_inference(image)
    if net.wait() == 0:
        output = net.extract_output
    return image, output

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame

if __name__ == "__main__":

    args = parse_args()

    net = VehicleDetection()
    net.load_model(args.model)

    cap, out, size = get_video(args)
    
    width = int(vehicle_cap.get(3))
    height = int(vehicle_cap.get(4))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Perform inference on the frame
        net.async_inference(p_frame)

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            print(result.shape)
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            out.write(frame)
        
            i += 1
            
            if i % 20 == 0:
                sys.stdout.write("Saving: " + int(i / 20).__str__() + " th second, ")
                sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break
    out.release()
    cap.release()