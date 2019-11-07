import copy
import threading as t

class Ring_buffer:
    def __init__(self, size):
        self.data_lock = t.Lock()
        self.data = [None for i in range(size)]
        self.size = size
        self.head = 0
        self.tail = 0
        self.head_over_tail = False


    def push_frame(self, frame):
        self.data_lock.acquire()

        self.data[self.tail] = copy.copy(frame)
        self.tail = (self.tail + 1) % self.size

        # manage when the tail overtake the head
        if self.tail == self.head:
            self.head_over_tail = True
        
        if self.head_over_tail and self.tail > self.head:
            self.head = self.tail

        self.data_lock.release()
    

    def pop_frame(self):
        self.data_lock.acquire()

        frame = self.data[self.head]
        self.data[self.head] = None

        self.head = (self.head + 1) % self.size

        # manage when the tail overtake the head
        if self.head_over_tail:
            self.head_over_tail = False

        self.data_lock.release()

        return frame

