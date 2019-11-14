import cv2 as cv
import threading as t
import copy
import time
import numpy as np
import queue
import json
from remote_zone import Remote_zone
from frames_buffer import Ring_buffer


def dispatcher_thread(computing_center, device):
        previous_frame = None

        while(True):

            req = computing_center.get_frame_to_dispatched(device)

            if req is None or req == previous_frame:
                continue
            
            previous_frame = req

            raw_frame = req['raw_frame']
            resized_shape = req['resized_shape']

            frame = np.frombuffer(raw_frame, dtype='uint8')
            frame = np.reshape(frame, json.loads(resized_shape))

            computing_center.push_new_input_frame(device, frame)
            computing_center.dispatch_frame(frame, device)

        pass


class Computing_center:

    def __init__(self, buffer_size, original_buffer_size, ratio, names_file):
        self.original_buffer_size = original_buffer_size
        self.buffer_size = buffer_size
        self.ratio = ratio
        self.names_file = names_file
        self.zones = {}
        self.devices = []
        self.thread_devices = {}
        self.input_queues = {}

        self.last_original_frames_buffers = {}
        self.last_rendered_frames = {}

        self.original_frames_locks = {}

        self.frames_to_be_dispatched = {}
        self.frames_to_be_dispatched_lock = t.Lock()


    def set_frame_to_be_dispatched(self, device, frame):
        self.frames_to_be_dispatched_lock.acquire()

        self.frames_to_be_dispatched[device] = frame

        self.frames_to_be_dispatched_lock.release()

    def get_frame_to_dispatched(self, device):
        self.frames_to_be_dispatched_lock.acquire()

        try:
            frame = copy.deepcopy(self.frames_to_be_dispatched[device])
        except KeyError:
            frame = None

        self.frames_to_be_dispatched_lock.release()

        return frame

    def add_device(self, dev):
        self.devices.append(dev)

        for zone in self.last_rendered_frames.values():
            zone[dev] = None

        for zone in self.zones.values():
            zone.add_device(dev)

        self.last_original_frames_buffers[dev] = Ring_buffer(self.original_buffer_size)
        self.original_frames_locks[dev] = t.Lock()

        self.input_queues[dev] = Ring_buffer(self.original_buffer_size)
        self.thread_devices[dev] = t.Thread(target = dispatcher_thread, args=(self, dev))
        self.thread_devices[dev].start()


    def push_new_input_frame(self, device, frame):
        success, encoded_img = cv.imencode(".jpg", frame)

        if not success:
            return

        self.original_frames_locks[device].acquire()
        self.last_original_frames_buffers[device].push_frame(encoded_img)
        self.original_frames_locks[device].release()


    def get_devices(self):
        self.devices.sort()
        return self.devices


    def get_zones(self):
        return self.zones.keys()


    def add_zone(self, name, zone):
        self.zones[name] = Remote_zone(name, zone, self.buffer_size, self.ratio, self.names_file)
        self.last_rendered_frames[zone] = {}
    
    
    def get_rendered_frame(self, zone, device):
        while True:

            frame = self.zones[zone].get_last_rendered_frame(device)
            
            if frame is None:
                continue

            success, encoded_img = cv.imencode(".jpg", frame)

            if not success:
                continue

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' 
            + bytearray(encoded_img) + b'\r\n')


    def dispatch_frame(self, frame, device):
        for zone in self.zones.values():
            zone.push_frame(frame, device)


    def get_original_frame(self, device):
        while True:
            
            frame = self.last_original_frames_buffers[device].pop_frame()

            if frame is None:
                continue
           
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' 
            + bytearray(frame) + b'\r\n')

                    






