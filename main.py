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
from flask import Flask, Response, render_template
import threading as t

from streamer import Streamer
from frames import Frames_coll

app = Flask(__name__, template_folder='templates')
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

@app.route("/original_video_feed")
def original_video_feed():
    return Response(
        streamer.stream_src_frame(),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/localhost_video_feed")
def localhost_video_feed():
    return Response(
        streamer.stream_remote_frame(streamer.localhost),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/edgehost_video_feed")
def edgehost_video_feed():
    return Response(
        streamer.stream_remote_frame(streamer.edgehost),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index(): 
    return render_template('index.html')


if __name__ == "__main__":
    input_type, input_path, localhost, edgehost = args_parsing()

    frames_collections = Frames_coll()
    frames_collections.add_output_queue(localhost)
    frames_collections.add_output_queue(edgehost)

    streamer = Streamer(localhost, edgehost, input_type, input_path, frames_collections)
    streamer.start_frames_getter()

    app.run(host='0.0.0.0', port=5005)
