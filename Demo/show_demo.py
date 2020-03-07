import time
import cv2

if __name__ == "__main__":

    gst_udp_read = 'udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink"

    cap = cv2.VideoCapture(gst_udp_read, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print('VideoCapture not opened')
        exit(0)

    while True:
        ret,frame = cap.read()

        if not ret:
            print('empty frame')
            break

        cv2.imshow('receive', frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()
