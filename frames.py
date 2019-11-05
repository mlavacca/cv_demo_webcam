class Frames_coll:
    def __init__(self):
        self.input_queue = []
        self.output_queues = {}
    
    def add_output_queue(self, host):
        self.output_queues[host] = []

    def push_input_frame(self, frame):
        self.input_queue.append(frame)

    def pop_input_frame(self):
        length = len(self.input_queue)

        if length > 0:
            frame = self.input_queue.pop(length - 1)
            return frame, True
        
        return None , False


    def push_output_frame(self, frame, host):
        self.output_queues[host].append(frame)
        pass
    
    def pop_output_frame(self, host):
        length = len(self.output_queues[host])

        if length > 0:
            frame = self.output_queues[host].pop(length - 1)
            return frame, True
        
        return None , False
