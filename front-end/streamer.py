import cv2 as cv
import threading as t
import time
import requests
import numpy as np
import json
import copy
from flask import request

class Streamer:

    def __init__(self, localhost, edgehost, type, path, size):
        self.n_frames = 0
        self.current_frame_n = 0
        self.localhost = localhost
        self.edgehost = edgehost
        self.type = type
        self.inputPath = path
        self.current_frame = None
        self.size = size

        self.delay = 20
        self.ratio = 10

        self.yielders = {}

        self.label = None
        self.boxes = None

        # Load names of classes
        classesFile = "coco.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        print("Streamer initialized")


    def add_yielder(self, client):
        self.yielders[client] = Frame_yielder(self.size)


    def stream_src_frame(self):
        while True:
            frame = self.localhost_frames_buffer.pop_frame()

            if frame is not None:

                flag, encoded_img = cv.imencode(".jpg", frame)

                self.current_frame_n += 1
            
                if not flag:
                    continue                
                
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_img) + b'\r\n')
    

    def stream_remote_frame(self, host):
        while True:
            self.input_frame_lock.acquire()
            frame = self.edge_frames_buffer.pop_frame()
            if frame is None:
                continue
            self.input_frame_lock.release()

            framecopy = copy.deepcopy(frame)
            frame = self.compute_frame(framecopy, host)
            
            success, encoded_img = cv.imencode(".jpg", frame)
     
            if not success:
                continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_img) + b'\r\n')
        

    def compute_frame(self, frame, host):

        if self.n_frames % self.ratio == 0:
            shape = list(frame.shape)

            req = requests.get(url=host,
                    data=frame.tobytes(),
                    headers={
                        'Content-Type': 'application/octet-stream',
                        'shape': json.dumps(shape)},
                    verify=False)

            inferenceTime = float(req.headers.get('inferenceTime'))
            self.label = 'Inference time: %.2f ms' % (inferenceTime)

            self.boxes = req.json()

        if self.boxes is not None:
            for stat in self.boxes:
                self.drawPred(frame, stat['class'], stat['confidence'], stat['left'], stat['top'], stat['leftWidth'], stat['topHeight'])

            cv.putText(frame, self.label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        return frame
       

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
        label = '%.2f' % conf
        
        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1) 