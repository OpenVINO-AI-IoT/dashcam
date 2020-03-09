import mpi4py
import numpy as np
import cv2
import os
from argparse import ArgumentParser
from mpi4py.futures import MPIPoolExecutor
import NetworkService
from Vehicle.VehicleDetection import VehicleDetection
from Service.VideoCapture import VideoCapture

LIB_TEXT_DETECTION = "./libs/libmain.so"
LIB_GRAPH = "./libs/openvx_graph.so"
LIB_CPU_EXTENSION = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so"
LIB_ALPR = "./ALPR/libalpr.so"

if os.path.isfile(LIB_ALPR):
    from libs import libalpr
if os.path.isfile(LIB_GRAPH):
    from libs import openvx_graph as graph
if os.path.isfile(LIB_TEXT_DETECTION):
    from libs import libmain as text

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

def build_graph_merge_mode(idx, args):
    pass

def get_video_vehicle(args):
    video_capture = VideoCapture(args.video_vehicle)
    if args.save_video:
        video_capture.create_output()
    return video_capture.cap, video_capture.out

def get_video_alpr(args):
    video_capture = VideoCapture(args.video_alpr)
    if args.save_video:
        video_capture.create_output()
    return video_capture.cap, video_capture.out

def get_video_text(args):
    video_capture = VideoCapture(args.video_text)
    if args.save_video:
        video_capture.create_output()
    return video_capture.cap, video_capture.out

def process_network(net, frame, input_shape):
    image = cv2.resize(frame, input_shape[0:2])
    image = image.reshape(2,0,1)
    net.async_inference(image)
    net.wait()
    output = net.extract_output
    return image, output

def write_plates_to_image(frame, plates, placements):
    # there can be multiple plates for each placement based on confidence level, there can be None as well
    plates_array = np.array(plates, dtype=np.object)
    idxs = plates_array[0,1]
    placements_array = np.array(placements)
    regions = placements_array[0,idxs]
    for region in regions:
        cv2.rectangle(frame, region[0:2], region[2:4], (0,0,255), thickness=3)
    return frame

def mpi_function(idx, args):
    image = None
    video = None

    identifier = worker_identifier(idx, args.merge_mode)

    try:
        if identifier.index("Vehicle"):
            net = VehicleDetection()
            net.load_model(args.model, args.d, LIB_CPU_EXTENSION)

            vehicle_cap, vehicle_out = get_video_vehicle(args)
            while True:
                ret, frame = cap.read()
                if ret == True:
                    input_shape = net.get_input_shape()
                    image, output = process_network(net, frame, input_shape)
                    if args.save_video:
                        out.write(output)
                    elif args.real_time:
                        cv2.imshow(identifier, output)
                else:
                    break
            vehicle_out.release()
            vehicle_cap.release()

        if identifier.index("ALPR"):
            alpr_cap, alpr_out = get_video_alpr(args)

            alpr = libalpr.ALPRImageDetect()

            alpr.Attributes(os.path.abspath("./ALPR/alpr_config/runtime_data/gb.conf"), 
            "eu", "", os.path.abspath("./ALPR/alpr_config/runtime_data"))

            while True:
                alpr_ret, alpr_frame = alpr_cap.read()
                if alpr_ret == True:
                    alpr.SetFrame(alpr_frame)
                    plates = alpr.LicensePlate_Matches(10, [[182,358,345,410]], False) # (182,358,345,410)
                    key_pressed = cv2.waitKey(60)
                    print(plates)
                else:
                    break
            alpr_out.release()
            alpr_cap.release()

        if identifier.index("Text"):
            text_cap, text_out = get_video_text(args)
            while True:
                ret, text_frame = cap.read()
                if ret == True:
                    txt = libmain.TextDetection(text_frame, 
                    "/home/dashcam/Text/trained_classifierNM1.xml", 
                    "/home/dashcam/Text/trained_classifierNM2.xml")
                    txt.Run_Filters()
                    image = txt.Get_Image()
                    if args.save_video:
                        text_out.write(image)
                    elif args.real_time:
                        cv2.imshow(identifier, image)
                else:
                    break
            text_out.release()
            text_cap.release()

    except Exception as e:
        raise e

def main(args):
    with MPIPoolExecutor(max_workers=3) as executor:
        executor.map(mpi_function, range(3), [args]*3)

    if out is not None:
        out.release()
    if cap is not None:
        cap.release()

if __name__ == "__main__":
    print("Welcome to Dashcam executable !!!")
    main(build_arguments_parser().parse_args())