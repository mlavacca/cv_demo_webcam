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
from flask import Flask

from streamer import Streamer

app = Flask(__name__)
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
        hostname = "http://localhost:5000"

    return input_type, input_path, hostname

@app.route("/original_video_feed")
def original_video_feed():
    pass

@app.route("/manipulated_video_feed")
def manipulated_video_feed():
    pass

@app.route("/", methods=['GET'])
def index():
    html_code = '''
<!DOCTYPE html>
<html>
<body>

<video width="720" height="400" controls>
  <source src="/home/mattia/Desktop/run.mp4" type="video/mp4">
  <source src="movie.ogg" type="video/ogg">
  Your browser does not support the video tag.
</video>

</body>
</html>
'''   

    return html_code, 200


if __name__ == "__main__":
    input_type, input_path, host = args_parsing()
    #streamer = Streamer(host, input_type, input_path)

    nFrame = 0
    fps = 0

    app.run(host='0.0.0.0', port=5005, debug=False)
