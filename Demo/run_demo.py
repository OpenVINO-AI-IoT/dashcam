import time
import cv2

# with closing(VideoSequence("Demo_video.mp4")) as frames:
#     for idx, frame in enumerate(frames[100:]):
#         frame.save("frame{:04d}.jpg".format(idx))

# Cam properties
fps = 30.
frame_width = 640
frame_height = 480
# Create capture
cap = cv2.VideoCapture('filesrc location=Demo_video.mp4 ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(gst_src_sink, cv2.CAP_GSTREAMER)
print(cap.read())
# Set camera properties
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# cap.set(cv2.CAP_PROP_FPS, fps)

# Define the gstreamer sink
gst_str_rtp = "gst-launch-1.0 ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000"

# Check if cap is open
if cap.isOpened() is not True:
    print("Cannot open camera. Exiting.")
    quit()

# Create videowriter as a SHM sink
out = cv2.VideoWriter(gst_str_rtp, 0, fps, (frame_width, frame_height), True)

# Loop it
while True:
    # Get the frame
    ret, frame = cap.read()
    # Check
    if ret is True:
        # Flip frame
        frame = cv2.flip(frame, 1)
        # Write to SHM
        out.write(frame)
    else:
        print("Camera error.")
        break
        time.sleep(10)

cap.release()