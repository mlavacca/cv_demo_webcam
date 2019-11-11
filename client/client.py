import cv2 as cv
import requests
import json
import time

def main():
    host = "http://localhost:5005/post_frame"

    cap = cv.VideoCapture(0)

    device = "cam1" 
    newSize = 416

    hasFrame, frame = cap.read()
        
    if not hasFrame:
        print("Cam is closing...")
        cap.release()

    shape = frame.shape
    oldH = shape[0]
    oldW = shape[1]
    ratio = oldH/oldW

    newW = 0
    newH = 0

    if shape[0] > newSize or shape[1] > newSize:    
        if oldH > oldW:
            newH = newSize
            newW = int(newH/ratio)
        else:
            newW = newSize
            newH = int(ratio * newW)
    
    newShape = [newH, newW, shape[2]]

    while True:  
        frame = cv.resize(frame, (newW, newH))

        try:    
            requests.post(
                url=host,
                data=frame.tobytes(),
                headers={
                'Content-Type': 'application/octet-stream',
                'original_shape': json.dumps(shape),
                'resized_shape': json.dumps(newShape),
                'device': device
                },
                verify=False)
        except Exception as e:
            print(e)
            print("Server connection closed")
            exit(-1)

        hasFrame, frame = cap.read()
        
        if not hasFrame:
            print("Cam is closing...")
            cap.release()
            break
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nClient closing")
