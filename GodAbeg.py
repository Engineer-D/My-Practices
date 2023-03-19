import torch
import torchvision
import time
import numpy as np
import cv2
from PIL import Image
from threading import Thread
from torchvision import models, transforms

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True
        
coco_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
model.eval()

#img = Image.open('000000037777.jpg')
#transform = transforms.ToTensor()
#img = transform(img)

# Initialize video stream
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize frame rate calculation
#frame_rate_calc = 1
#freq = cv2.getTickFrequency()

# Initialize video stream
#videostream = VideoStream(framerate=30).start()
#time.sleep(1)

# Create window
#cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

with torch.no_grad():
    while True:
        # Start timer (for calculating frame rate)
        #t1 = cv2.getTickCount()
        
        # Grab frame from video stream
        #frame1 = videostream.read()
        
        # Acquire frame and resize to expected shape [1xHxWx3]
        #frame = frame1.copy()
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        # preprocess
        #input_tensor = preprocess(frame_resized)

        # create a mini-batch as expected by the model
        #input_batch = input_tensor.unsqueeze(0)
        
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        #input_batch = input_tensor.unsqueeze(0)

        # run model
        pred = model([input_tensor])
        #time.sleep(20)
        print(pred)
        
        # Retrieve detection results
        bboxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']
        num = torch.argwhere(scores > 0.3).shape[0]
        print(labels)
        break
        
        for i in range(num):
            x1,y1,x2,y2 = bboxes[i].numpy().astype('int')
            class_name = coco_names[labels.numpy()[i] - 1]
            img = cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0), 1)
            img = cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
            print(f"Object {i}: {class_name}")
            
        # Draw framerate in corner of frame
        #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        #display
        #cv2.imshow('Object detector', frame)
        
        # Calculate framerate
        #t2 = cv2.getTickCount()
        #time1 = (t2-t1)/freq
        #frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

# Clean up
cv2.destroyAllWindows()
#videostream.stop()
cap.release()