FROM python:3.7

ADD yolo_server.py /
ADD getModels.sh /

RUN pip install Flask 
RUN pip install opencv-python
RUN pip install prometheus_client
RUN pip install PyYAML

RUN ./getModels.sh

EXPOSE 5005

CMD ["python", "main.py", "-f", "config.yml"]

