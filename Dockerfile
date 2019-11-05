FROM python:3.7

ADD yolo_server.py /
ADD getModels.sh /

RUN pip install Flask 
RUN pip install opencv-python
RUN pip install pprint
RUN pip install prometheus_client

RUN ./getModels.sh

EXPOSE 5000

CMD ["python", "./yolo_server.py"]

