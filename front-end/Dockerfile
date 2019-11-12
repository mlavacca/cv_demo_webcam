FROM python:3.7

RUN pip install Flask 
RUN pip install opencv-python
RUN pip install prometheus_client
RUN pip install PyYAML
RUN pip install requests

RUN wget https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true -O ./coco.names

RUN mkdir templates
ADD ./*.py /
ADD ./config.yml /
ADD ./templates/ /templates

EXPOSE 5005

CMD ["python", "main.py", "-f", "config.yml"]

