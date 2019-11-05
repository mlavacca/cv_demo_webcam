import requests
import threading as t
import json
import cv2 as cv
import time
from datetime import datetime
from frames_buffer import Ring_buffer


class Remote_zone:

    def __init__(self, name, dest, buffer_size, ratio, names_file):
        self.name = name
        self.dest = dest
        self.buffer_size = buffer_size

        self.in_frame_bridges = {}
        self.out_frame_bridges = {}

        self.rendering_threads = {}

        self.ratio = ratio
        self.n_computed_frames = {}
        
        # last objects handled by the current zone
        self.last_boxes = {}
        self.last_inference_times = {}
        self.last_labels = {}
        self.metadata_locks = {}

        # file containing the names of the objects recognized by opencv
        self.names_file = names_file
        with open(names_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')


    def add_device(self, device):
        self.in_frame_bridges[device] = Ring_buffer(self.buffer_size)
        self.out_frame_bridges[device] = Ring_buffer(self.buffer_size)

        self.last_boxes[device] = {}

        self.last_labels[device] = {}
        self.last_inference_times[device] = {}
        self.n_computed_frames[device] = 0

        th = t.Thread(target=self.rendering_loop, args=(device, ))
        th.start()
        self.rendering_threads[device] = th


    # send frame to the remote server
    def send_frame(self, frame, dev):
        shape = list(frame.shape)

        start = datetime.now()
        req = requests.get(url=self.dest,
                    data=frame.tobytes(),
                    headers={
                        'Connection': 'Keep-Alive',
                        'Content-Type': 'application/octet-stream',
                        'shape': json.dumps(shape)},
                    verify=False)
        end = datetime.now()
        buffer = self.name + ": " + str(end-start)
        print(buffer)

        if req.status_code != 200:
            return -1

        self.last_boxes[dev] = req.json()
        self.last_inference_times[dev] = float(req.headers.get('inferenceTime'))
        self.last_labels[dev] = 'Inference time: %.2f ms' % (self.last_inference_times[dev])

        return 0



    # render the current frame
    def render_frame(self, frame, dev):

        if self.last_boxes[dev] is None or self.n_computed_frames[dev] % self.ratio == 0:
            if self.send_frame(frame, dev) != 0:
                return None

        for stat in self.last_boxes[dev]:
            self.drawPred(frame, stat['class'], stat['confidence'], stat['left'], stat['top'], stat['leftWidth'], stat['topHeight'])

        cv.putText(frame, self.last_labels[dev], (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        self.n_computed_frames[dev] += 1
        
        return frame


    def rendering_loop(self, device):

        while True:
            start = time.time() * 1000
            frame = self.in_frame_bridges[device].pop_frame()

            if frame is None:
                continue
            
            frame = self.render_frame(frame, device)
            
            if frame is None:
                continue
            
            req_time = (time.time() * 1000) - start
            self.out_frame_bridges[device].push_frame(frame)


    # push the new frame in the input buffer
    def push_frame(self, frame, device):
        self.in_frame_bridges[device].push_frame(frame)


    # pop the first object from the input buffer
    def pop_frame(self, device):
        return self.in_frame_bridges[device].pop_frame()


    # return the last available frame
    def get_last_rendered_frame(self, device):
        frame = self.out_frame_bridges[device].pop_frame()

        return frame


    # draw the boxes on the frame
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
