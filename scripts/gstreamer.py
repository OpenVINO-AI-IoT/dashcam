import sys
sys.path.append("../")
sys.path.append("/home/openalpr/openalpr/src/bindings/python")
from Service.LicensePlateIdentifier import LicensePlateIdentifier
from Service.GStreamer import CLI_Main
from openalpr import Alpr
from Service.VideoCapture import VideoCapture
import argparse
import cv2

