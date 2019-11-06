import cv2 as cv
import threading as t
import copy
from remote_zone import Remote_zone
from frames_buffer import Ring_buffer


class Computing_center:

    def __init__(self, buffer_size, ratio, names_file):
        self.buffer_size = buffer_size
        self.ratio = ratio
        self.names_file = names_file
        self.zones = {}
        self.devices = []

        self.last_original_frames = {}
        self.last_rendered_frames = {}

        self.original_frames_locks = {}
        self.rendered_frames_locks = {}


    def add_device(self, dev):
        self.devices.append(dev)

        for zone in self.last_rendered_frames.values():
            zone[dev] = None

        self.original_frames_locks[dev] = t.Lock()
        self.rendered_frames_locks[dev] = t.Lock()


    def add_zone(self, zone):
        self.zones[zone] = Remote_zone(zone, self.buffer_size, self.ratio, self.names_file)
        self.last_rendered_frames[zone] = {}
    
    
    def get_rendered_frame(self, zone, device):
        while True:
            self.rendered_frames_locks[device].acquire()

            frame = self.zones[zone].get_last_rendered_frame()
            
            if frame is None:
                self.rendered_frames_locks[device].release() 
                continue

            success, encoded_img = cv.imencode(".jpg", frame)

            self.rendered_frames_locks[device].release()   

            if not success:
                continue

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' 
            + bytearray(encoded_img) + b'\r\n')


    def distribute_frame(self, frame, device):
        self.original_frames_locks[device].acquire()
        self.last_original_frames[device] = copy.copy(frame)
        self.original_frames_locks[device].release()

        for zone in self.zones.values():
            zone.push_frame(frame, device)


    def get_original_frame(self, device):
        while True:
            self.original_frames_locks[device].acquire()

            if self.last_original_frames[device] is None:
                self.original_frames_locks[device].release()
                continue

            success, encoded_img = cv.imencode(".jpg", self.last_original_frames[device])

            self.original_frames_locks[device].release()
            
            if not success:
                continue

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' 
            + bytearray(encoded_img) + b'\r\n')

                    





