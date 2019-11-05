import cv2 as cv
import threading as t
import time
import requests
import numpy as np
import json
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
                frame = self.compute_frame(frame)

            flag, encoded_img = cv.imencode(".jpg", frame)

            if not flag:
                continue
            
            self.n_frames += 1
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_img) + b'\r\n')


    def compute_frame(self, frame):
        shape = frame.shape

        # create the POST request embedding the frame
        req = requests.get(url=self.host,
                    data=frame.tobytes(),
                    headers={
                        'Content-Type': 'application/octet-stream',
                        'shape': json.dumps(shape)},
                    verify=False)

        recvFrame = np.frombuffer(req.content, dtype='uint8')

        try:
            recvFrame = np.reshape(recvFrame, [480, 640, 3])
        except Exception:
            print(req.status_code)
            print(req.text)

        return recvFrame
         