import cv2
import sys
sys.path.append("../")
from NetworkService import NetworkService

class VehicleDetection(NetworkService):

    def __init__(self, delay=-1, confidence_threshold=0.4, thickness=3, color=(0,0,255)):
        super(VehicleDetection, self).__init__()
        self.delay = delay
        self.confidence_threshold = confidence_threshold
        self.thickness = thickness
        self.color = color
        self.bounding_boxes = []

    def obtain_frame(self, cap):
        frame_resize = None
        try:
            ret, frame = cap.read()
            if ret == True:
                frame_resize = cv2.resize(frame,(672,384))
        except Exception as e:
            print(e.args)
            
        return frame_resize

    def real_time_update(self, frame, xmin, y_min, x_max, y_max):
        cv2.rectangle(frame, (xmin, y_min), (x_max, y_max), self.color, self.thickness)
        return frame

    def get_bounding_boxes(self, output, frame):
        rt_frame = frame.copy()
        for detections in output[0][0]:
            image_id, label, conf, x_min, y_min, x_max, y_max = tuple(detections)
            if conf > self.confidence_threshold:
                rt_frame = self.real_time_update(rt_frame, x_min, y_min, x_max, y_max)
                self.bounding_boxes.append((x_min, y_min, x_max, y_max))

        return rt_frame

