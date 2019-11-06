#!/usr/bin/python3

import argparse
import sys
import numpy as np
import os.path
import requests
import http
import json
from datetime import datetime
from pprint import pprint
import codecs
from flask import Flask, Response, render_template, request
import threading as t

from computing_center import Computing_center
from streamer import Streamer

app = Flask(__name__, template_folder='../templates')
app.secret_key = "secret_key"

def args_parsing():
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--video', help='Path to video file.')
    parser.add_argument('--host', help='Server host')
    args = parser.parse_args()
    
    if args.video:
        input_path = args.video
        if not os.path.isfile(input_path):
            print("error")
            exit(-1)
        input_type = "video"
    else:
        input_path = ""
        input_type = "cam"
        
    if args.host:
        edgehost = args.host
    else:
        edgehost = None
    
    localhost = "http://localhost:5000/processframe"
    
    return input_type, input_path, localhost, edgehost


@app.route("/post_frame", methods=['POST'])
def post_frame():
    try:
        r = request.get_data()
        device = request.headers.get('device')
        shape = request.headers.get('shape')

        frame = np.frombuffer(r, dtype='uint8')
        frame = np.reshape(frame, json.loads(shape))

        if device not in computing_center.devices:
            computing_center.add_device(device)

        computing_center.distribute_frame(frame, device)
    except Exception as e:
        print(e)
        return "BAD REQUEST", 400
    
    return "OK", 200


@app.route("/original_video_feed", )
def original_video_feed():
    device = request.args.get('device')

    return Response(
        computing_center.get_original_frame(device),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )



@app.route("/remote_video_feed")
def remote_video_feed():
    device = request.args.get('device')
    zone = request.args.get('zone')

    return Response(
        computing_center.get_rendered_frame(zone, device),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index(): 
    return render_template('index.html')


if __name__ == "__main__":

    computing_center = Computing_center(5, 5, 'coco.names')

    computing_center.add_zone('http://localhost:5000/processframe')
    computing_center.add_zone('http://130.192.225.63:32255/processframe')

    app.run(host='0.0.0.0', port=5005)
