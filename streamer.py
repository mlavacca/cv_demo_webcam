import cv2 as cv


class Streamer:

    def __init__(self, host, type, path):
        self.host = host
        self.type = type
        self.inputPath = path

        if type == "video":
            self.cap = cv.VideoCapture(self.inputPath)
            self.outputFile = self.inputPath + "_cv.avi"
            self.vid_writer = cv.VideoWriter(self.outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, 
                              (round(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    
        if type == "cam":
            self.cap = cv.VideoCapture(0)
            self.outputFile = "cam_video_cv.avi"
            self.vid_writer = cv.VideoWriter(self.outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30,
                              (round(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        print("Streamer initialized")


    def stream_src():
        pass


    def stream_dest():
        pass


    def run():
        '''
        while cv.waitKey(1) < 0:
            itStart = datetime.now()
    
            # get frame from the video
            hasFrame, frame = cap.read()

            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                print("Output file is stored as ", outputFile)
                cv.waitKey(3000)
                # Release device
                cap.release()
                break

            shape = frame.shape


            # create the POST request embedding the frame
            req = requests.get(url=urlDest,
                    data=frame.tobytes(),
                    headers={'Content-Type': 'application/octet-stream'},
                    verify=False)

            inference = req.headers['inferenceTime']
            serverTime = req.headers['serverTime']

            recvFrame = np.frombuffer(req.content, dtype='uint8')
            try:
                recvFrame = np.reshape(recvFrame, shape)
            except Exception:
                print(req.status_code)
                print(req.text)
                break

            # Write the frame with the detection boxes
            if (args.image):
                cv.imwrite(outputFile, recvFrame.astype(np.uint8))
            else:
                vid_writer.write(recvFrame.astype(np.uint8))

            cv.imshow(winName, recvFrame)

            nFrame += 1
            itTime = datetime.now() - itStart
            newfps = (1/itTime.total_seconds())
            fps = ((newfps) + (fps * (nFrame - 1)))/(nFrame)

            print("%-20s: %.2f" %("framerate", fps))
            print("%-20s: %.2f" %("frame time", itTime.total_seconds() * 1000))
            print("%-20s: %.2f" %("Inference time", float(inference)))
            print("%-20s: %.2f" %("Server time", float(serverTime)))
            print()
        '''
         