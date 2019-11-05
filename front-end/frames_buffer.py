import copy
import threading as t

class Ring_buffer:
    
    def __init__(self, size):
        self.data_lock = t.Lock()
        self.data = [None for i in range(size)]
        self.size = size
        self.n_elements = 0
        self.head = 0
        self.tail = 0
        self.tail_over_head = False
        self.cv = t.Condition()


    def push_frame(self, frame):
        self.data_lock.acquire()

        # push the new frame
        self.data[self.tail] = frame
        self.tail = (self.tail + 1) % self.size

        # increase the counter of frames
        self.n_elements += 1
        if self.n_elements > 5:
            self.n_elements = 5

        # notify that there is a new element
        with self.cv:
            self.cv.notifyAll()

        # the tail has just reached the head
        if self.tail == self.head:
            self.tail_over_head = True
        
        # manage when the tail overtakes the head
        if self.tail_over_head and self.tail > self.head:
            self.head = self.tail

        self.data_lock.release()
    

    def pop_frame(self):

        # if the number of frames is 0, wait for a new frame
        if self.n_elements == 0:
            with self.cv:
                self.cv.wait()

        self.data_lock.acquire()

        # pop the oldest frame
        frame = self.data[self.head]
        self.data[self.head] = None
        self.head = (self.head + 1) % self.size

        # decrease the frame counter
        self.n_elements -= 1

        if self.tail_over_head:
            self.tail_over_head = False

        self.data_lock.release()

        return frame

