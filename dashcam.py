import mpi4py
import numpy as np
import cv2
import os, sys
sys.path.append("/home/openalpr/openalpr/src/bindings/python")
from openalpr import Alpr
from argparse import ArgumentParser
from mpi4py.futures import MPIPoolExecutor
import NetworkService
from time import sleep, time
from Vehicle import VehicleDetection
from Service.VideoCapture import VideoCapture

LIB_TEXT_DETECTION = "./libs/text_detection.so"
LIB_GRAPH = "./libs/openvx_graph.so"
LIB_CPU_EXTENSION = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so"
LIB_ALPR = "./ALPR/libalpr.so"

if os.path.isfile(LIB_ALPR):
    from ALPR import libalpr
if os.path.isfile(LIB_GRAPH):
    from libs import openvx_graph as graph
if os.path.isfile(LIB_TEXT_DETECTION):
    from libs import text_detection as text

frame = cv2.imread("image.png")

alpr_detect = libalpr.ALPRImageDetect(frame)
alpr_detect.Attributes(os.path.join(os.path.abspath("./ALPR"), libalpr.config),
    libalpr.region, libalpr.country, 
    "/home/dashcam/ALPR/runtime_data")
    
def build_arguments_parser():
    parser = ArgumentParser(description='Dashcam project for home, work and holiday modes', allow_abbrev=False)
    parser.add_argument(
        '-mode', '--mode',
        help='mode of the executable',
        type=str,
        default="holiday",
        required=True
    )
    parser.add_argument(
        '-v', '--video',
        help='video path',
        type=str,
        default="assets/sample_video.mp4",
        required=True
    )
    parser.add_argument(
        '-sf', '--scale_factor',
        help='scale factor to for scaled tampering',
        type=float,
        default=1.0,
        required=True
    )
    parser.add_argument(
        '-rt', '--real_time',
        help='to show the demo in real time',
        type=bool,
        default=False,
        required=True
    )
    parser.add_argument(
        '-save', '--save_video',
        help='to save the video to file system',
        type=bool,
        default=True,
        required=True
    )
    parser.add_argument(
        '-merge', '--merge_mode',
        help='to show the modes of operation of the demo video (home,holiday,work)',
        type=bool,
        default=False,
        required=True
    )
    parser.add_argument(
        '-d', '--device',
        help='the device to execute',
        type=str,
        default="CPU",
        required=True
    )
    parser.add_argument(
        '-m', '--model',
        help="the deep learning model",
        type=str,
        default="./Vehicle/model/model.xml",
        required=True
    )
    # optimizations parameters

    return parser

def worker_identifier(worker_index, worker_mode):
    if(worker_index == 0 and worker_mode == False):
        return "Worker thread for Text detection"
    elif(worker_index == 1 and worker_mode == False):
        return "Worker thread for Vehicle detection"
    elif(worker_index == 2 and worker_mode == False):
        return "Worker thread for ALPR"
    elif(worker_index == 0 and worker_mode == True):
        return "Worker thread for Home Mode"
    elif(worker_index == 1 and worker_mode == True):
        return "Worker thread for Holiday Mode"
    elif(worker_index == 2 and worker_mode == True):
        return "Worker thread for Work Mode"

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

def build_graph_merge_mode(idx, args):
    pass

def get_video(args):
    video_capture = VideoCapture(args.video)
    if args.save_video:
        video_capture.create_output()
    return video_capture.cap, video_capture.out

def process_network(net, frame, input_shape):
    image = cv2.resize(frame, input_shape[0:2])
    image = image.reshape(2,0,1)
    net.async_inference(image)
    if net.wait() == 0:
        output = net.extract_output()

    return output

def write_plates_to_image(frame, plates, placements):
    # there can be multiple plates for each placement based on confidence level, there can be None as well
    plates_array = np.array(plates, dtype=np.object)
    idxs = plates_array[0,1]
    placements_array = np.array(placements)
    regions = placements_array[0,idxs]
    for region in regions:
        cv2.rectangle(frame, region[0:2], region[2:4], (0,0,255), thickness=3)
    return frame

def combine_images():
    

def mpi_function(idx, args, frame, cap, state):
    image = None
    video = None

    identifier = worker_identifier(idx, args.merge_mode)

    try:
        if identifier.index("Vehicle"):
            net = VehicleDetection()
            net.load_model(args.model, args.d, LIB_CPU_EXTENSION)

            input_shape = net.get_input_shape()

            output = process_network(net, frame, input_shape)
            image = draw_boxes(state.clone(), output, args, width, height)
            return image

        if identifier.index("ALPR"):
            region = args.region if args.region else alpr.region
            country = args.country if args.country else alpr.country

            alpr_detect = alpr.ALPRImageDetect(frame, 
            os.path.join([os.path.abspath("./ALPR"), alpr.config]),
            region, country)
            plates = alpr_detect.LicensePlate_Matches()
            placements = alpr_detect.Placements()
            if args.save_video:
                output = write_plates_to_image(frame, plates, placements)
                out.write(output)
            elif args.real_time:
                output = write_plates_to_image(frame, plates, placements)
                cv2.imshow(identifier, output)
            return image

        if identifier.index("Text"):
            text.Run_Filters()
            state = txt.Groups_Draw(state)
            return image

    except Exception as e:
        raise e

def main(args):
    cap, out = get_video(args)
    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    color_state_image = np.zeros((width, height, 3))

    alpr = Alpr("eu", "./ALPR/alpr_config/runtime_data/gb.conf", 
    "./ALPR/alpr_config/runtime_data")
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

    lib_merges = []

    while True:
        ret, frame = cap.read()
        if ret == True:
            with MPIPoolExecutor(max_workers=3) as executor:
                results = []
                for result in executor.map(mpi_function, 
                range(3), [args]*3, [frame]*3, [cap]*3, 
                [color_state_image]*3):
                    results.append(result)
            if args.save_video:
                out.write(output)
            if args.real_time:
                cv2.imshow(identifier, output)
        else:
            break

    if out is not None:
        out.release()
    if cap is not None:
        cap.release()

if __name__ == "__main__":
    print("Welcome to Dashcam executable !!!")
    main(build_arguments_parser().parse_args())