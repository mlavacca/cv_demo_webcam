import requests
import json
import cv2 as cv
from frames_buffer import Ring_buffer


class Remote_zone:

    def __init__(self, dest, buffer_size, ratio, names_file):
        self.dest = dest
        self.frame_bridge = Ring_buffer(buffer_size)

        self.ratio = ratio
        self.n_computed_frame = 0
        
        # last objects handled by the current zone
        self.last_frames = {}
        self.last_rendered_frames = {}
        self.last_boxes = {}
        self.last_inference_times = {}
        self.last_labels = {}

        # file containing the names of the objects recognized by opencv
        self.names_file = names_file
        with open(names_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')


    # send frame to the remote server
    def send_frame(self, frame, dev):
        shape = list(frame.shape)

        req = requests.get(url=self.dest,
                    data=frame.tobytes(),
                    headers={
                        'Content-Type': 'application/octet-stream',
                        'shape': json.dumps(shape)},
                    verify=False)

        self.last_boxes[dev] = req.json()
        self.last_inference_times[dev] = float(req.headers.get('inferenceTime'))
        self.last_labels[dev] = 'Inference time: %.2f ms' % (self.last_inference_times[dev])


    # render the current frame
    def render_frame(self, frame, dev):

        if self.last_boxes is None or self.n_computed_frame % self.ratio == 0:
            self.send_frame(frame, dev)

        for stat in self.last_boxes[dev]:
            self.drawPred(frame, stat['class'], stat['confidence'], stat['left'], stat['top'], stat['leftWidth'], stat['topHeight'])

        cv.putText(frame, self.last_labels[dev], (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        self.n_computed_frame += 1
        
        return frame


    # push the new frame in the output buffer
    def push_frame(self, frame, device):
        self.frame_bridge.push_frame({'dev': device, 'frame': frame})


    def pop_frame(self, device):
        return self.frame_bridge.pop_frame()


    def get_last_rendered_frame(self):
        data = self.frame_bridge.pop_frame()

        frame = data['frame']
        dev = data['dev']

        if frame is not None:
            self.last_rendered_frames[dev] = self.render_frame(frame, dev)
        
        return self.last_rendered_frames[dev]


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
