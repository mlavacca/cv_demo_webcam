FROM python:3.7

ADD yolo_server.py /

RUN pip install Flask 
RUN pip install opencv-python
RUN pip install PyYAML

RUN wget https://pjreddie.com/media/files/yolov3-tiny.weights
RUN wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true -O ./yolov3-tiny.cfg

RUN wget https://pjreddie.com/media/files/yolov3.weights
RUN wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true -O ./yolov3.cfg

EXPOSE 5000

CMD ["python", "yolo_server.py"]

