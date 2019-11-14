import copy
import threading as t

class Ring_buffer:
    
    def __init__(self, size):
        self.data_lock = t.Lock()
        self.data = [None for i in range(size)]
        self.size = size
        self.head = 0
        self.tail = 0
        self.tail_over_head = False


    def push_frame(self, frame):
        self.data_lock.acquire()

        self.data[self.tail] = copy.deepcopy(frame)
        self.tail = (self.tail + 1) % self.size

        # the tail has just reached the head
        if self.tail == self.head:
            self.tail_over_head = True
        
        # manage when the tail overtakes the head
        if self.tail_over_head and self.tail > self.head:
            self.head = self.tail

        self.data_lock.release()
    

    def pop_frame(self):
        self.data_lock.acquire()

        frame = self.data[self.head]
        self.data[self.head] = None

        self.head = (self.head + 1) % self.size

        if self.tail_over_head:
            self.tail_over_head = False

        self.data_lock.release()

        return frame

    def get_last_frame(self):
        self.data_lock.acquire()

        frame = self.data[self.head]

        self.data_lock.release()

        return frame