import cv2 as cv
import requests
import json

def main():
    host = "http://localhost:5005/post_frame"
    cap = cv.VideoCapture(0)
    device = "cam0"   

    while True:    
        hasFrame, frame = cap.read()
            
        shape = frame.shape

        if not hasFrame:
            print("Cam is closing...")
            cap.release()
            break

        try:    
            requests.post(
                url=host,
                data=frame.tobytes(),
                headers={
                'Content-Type': 'application/octet-stream',
                'shape': json.dumps(shape),
                'device': device
                },
                verify=False)

        except Exception:
            print("ERR - client connection error")
            exit(-1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nClient closing")
