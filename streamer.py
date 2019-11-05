import cv2 as cv
import threading as t
import time
import requests
import numpy as np
import json
from flask import request
from frames import Frames_coll

class Streamer:

    def __init__(self, localhost, edgehost, type, path, frames_coll):
        self.n_frames = 0
        self.ratio = 1
        self.localhost = localhost
        self.edgehost = edgehost
        self.type = type
        self.inputPath = path
        self.frames_coll = frames_coll
        self.frames_coll_lock = t.Lock()
        self.shape = (0, 0)

        if type == "video":
            self.cap = cv.VideoCapture(self.inputPath)
            self.outputFile = self.inputPath + "_cv.avi"
            self.vid_writer = cv.VideoWriter(self.outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, 
                              (round(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    
        if type == "cam":
            self.cap = cv.VideoCapture(0)

        # Load names of classes
        classesFile = "coco.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        print("Streamer initialized")


    def start_frames_getter(self):
        self.frame_getter_thread = t.Thread(target=self.get_frames, args=())
        self.frame_getter_thread.start()


    def get_shape(self):
        return self.shape[0], self.shape[1]


    def get_frames(self):
        while True:    
            # get frame from the video
            hasFrame, frame = self.cap.read()
            self.shape = frame.shape

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", self.outputFile)
                cv.waitKey(3000)
                # Release device
                self.cap.release()
                break
            
            self.frames_coll_lock.acquire()
            self.frames_coll.push_input_frame(frame)
            self.frames_coll.push_output_frame(frame, self.localhost)
            self.frames_coll.push_output_frame(frame, self.edgehost)
            self.frames_coll_lock.release()

    def stream_src_frame(self):
        while True:
            self.frames_coll_lock.acquire()
            frame, flag = self.frames_coll.pop_input_frame()
            self.frames_coll_lock.release()

            if not flag:
                continue
            
            flag, encoded_img = cv.imencode(".jpg", frame)

            if not flag:
                continue
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_img) + b'\r\n')
    
    def stream_remote_frame(self, host):
        while True:
            self.frames_coll_lock.acquire()
            frame, flag = self.frames_coll.pop_output_frame(host)
            self.frames_coll_lock.release()

            if not flag:
                continue
            
            if self.n_frames % self.ratio == 0:
                frame = self.compute_frame(frame, host)

            flag, encoded_img = cv.imencode(".jpg", frame)

            if not flag:
                continue
            
            self.n_frames += 1
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_img) + b'\r\n')


    def compute_frame(self, frame, host):
        shape = list(frame.shape)

        # create the POST request embedding the frame
        req = requests.get(url=host,
                    data=frame.tobytes(),
                    headers={
                        'Content-Type': 'application/octet-stream',
                        'shape': json.dumps(shape)},
                    verify=False)

        inferenceTime = float(req.headers.get('inferenceTime'))
        label = 'Inference time: %.2f ms' % (inferenceTime)

        data = req.json()
        for stat in data:
            self.drawPred(frame, stat['class'], stat['confidence'], stat['left'], stat['top'], stat['leftWidth'], stat['topHeight'])

        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

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