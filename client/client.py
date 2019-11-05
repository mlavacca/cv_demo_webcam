#!/usr/bin/python3.7

import cv2 as cv
import requests
import json
import time
import threading

frame = None
newShape = None
oldH = None
oldW = None
ratio = None
shape = None

newW = 0
newH = 0

cap = None

dest_threads = []
closing_flag = False
lock = None


def send_frames(dest):
    global lock, shape, newShape
    global closing_flag

    device = "cam1"

    while closing_flag is False:
        try:    
            if frame is  None:
                continue
            
            lock.acquire()
            f = frame.tobytes()
            lock.release()

            requests.post(
                url=dest,
                data=f,
                headers={
                'Connection': 'Keep-Alive',
                'Content-Type': 'application/octet-stream',
                'original_shape': json.dumps(shape),
                'resized_shape': json.dumps(newShape),
                'device': device
                },
                verify=False)

        except Exception as e:
            print("Can't connect to ", dest)
            time.sleep(2)


def main():
    global frame
    global newShape, shape
    global oldH, oldW, newH, newW, ratio
    global lock
    global cap

    hosts = [
        #"http://130.192.225.63:30389/post_frame",
        #"http://34.65.232.227:32203/post_frame",
        "http://localhost:5005/post_frame"
        ]

    for host in hosts:
        dest_threads.append(threading.Thread(target=send_frames, args=(host, )))

    lock = threading.Lock()

    cap = cv.VideoCapture(0)

    newSize = 416

    hasFrame, f = cap.read()
        
    if not hasFrame:
        print("Cam is closing...")
        cap.release()

    shape = f.shape
    oldH = shape[0]
    oldW = shape[1]
    ratio = oldH/oldW

    if shape[0] > newSize or shape[1] > newSize:    
        if oldH > oldW:
            newH = newSize
            newW = int(newH/ratio)
        else:
            newW = newSize
            newH = int(ratio * newW)
    
    newShape = [newH, newW, shape[2]]

    for t in dest_threads:
        t.start()

    while True:  
    
        hasFrame, frame_buffer = cap.read()

        lock.acquire()
        frame=frame_buffer

        if not hasFrame:
            print("Cam is closing...")
            cap.release()
            break
        
        frame = cv.resize(frame, (newW, newH))
        lock.release()

        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        closing_flag = True
        print("\nClient closing")
        cap.release()
        for t in dest_threads:
            t.join()
