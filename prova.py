import copy
import threading as t

class Frames_buffer:
    def __init__(self, size):
        self.data_lock = t.Lock()
        self.data = [None for i in range(size)]
        self.size = size
        self.head = 0
        self.tail = 0

    def push_frame(self, frame):
        self.data_lock.acquire()
        self.data[self.head] = frame
        self.head = (self.head + 1) % self.size
        self.data_lock.release()
        pass
    
    def pop_frame(self):
        self.data_lock.acquire()
        frame = self.data[self.tail]
        self.data[self.tail] = None
        self.tail = (self.tail + 1) % self.size
        
        self.data_lock.release()

        return frame

f = Frames_buffer(5)

for i in range(0, 10):
    f.push_frame(i)
    f.push_frame(i + 10)
    print(f.data)
    n = f.pop_frame()
    print(n)
    print(f.data)