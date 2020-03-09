import mpi4py
import numpy as np
import cv2
import os
from argparse import ArgumentParser
from mpi4py.futures import MPIPoolExecutor
import NetworkService
from Vehicle.VehicleDetection import VehicleDetection
from Service.VideoCapture import VideoCapture

LIB_CPU_EXTENSION = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so"

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

def main(args):
    

if __name__ == "__main__":
    print("Welcome to Dashcam executable !!!")
    main(build_arguments_parser().parse_args())