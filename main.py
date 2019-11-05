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
        hostname = "http://" + args.host + ":5000"
    else:
        hostname = "http://localhost:5000/processframe"

    return input_type, input_path, hostname

@app.route("/original_video_feed")
def original_video_feed():

    return Response(
        src_streamer.stream_src_frame(),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/localhost_video_feed")
def localhost_video_feed():
    return Response(
        src_streamer.stream_remote_frame(src_streamer.host),
        mimetype = "multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/", methods=['GET'])
def index(): 
    return render_template('index.html')


if __name__ == "__main__":
    input_type, input_path, host = args_parsing()

    frames_collections = Frames_coll()
    frames_collections.add_output_queue(host)

    src_streamer = Streamer(host, input_type, input_path, frames_collections)
    src_streamer.start_frames_getter()


    app.run(host='0.0.0.0', port=5005, debug=False)
