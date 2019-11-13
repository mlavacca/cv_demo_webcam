#!/usr/bin/python3

import argparse
import sys
import numpy as np
import os.path
import requests
import http
import json
from flask import Flask, Response, render_template, request
import threading as t
import yaml

from computing_center import Computing_center

app = Flask(__name__, template_folder='templates')
app.secret_key = "secret_key"


# args parsing for configuration file
def args_parsing():
    parser = argparse.ArgumentParser(description='Middle point that acts as frame aggregator and dispatcher')
    parser.add_argument('-f', '--file', default="config.yml" , help='Path to configuration file')
    args = parser.parse_args()
        
    if args.file:
        config_path = args.file
        if not os.path.isfile(config_path):
            print("error - config file not found")
            exit(-1)
    
    return config_path


@app.route("/post_frame", methods=['POST'])
def post_frame():
    try:
        r = request.get_data()
        device = request.headers.get('device')
        original_shape = request.headers.get('original_shape')
        resized_shape = request.headers.get('resized_shape')

        frame = np.frombuffer(r, dtype='uint8')
        frame = np.reshape(frame, json.loads(resized_shape))

        if device not in computing_center.devices:
            computing_center.add_device(device)

        computing_center.dispatch_frame(frame, device)
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

    return render_template('index.html', 
            devices=computing_center.get_devices(),
            zones=computing_center.get_zones()
    )


if __name__ == "__main__":

    args = args_parsing()

    conf_file = open(args, "r")
    buffer = conf_file.read()
    conf = yaml.load(buffer)

    computing_center = Computing_center(conf['buffer_size'], conf['original_buffer_size'], conf['ratio'], conf['objects_names'])

    for zone in conf['zones']:
        computing_center.add_zone(zone['name'], zone['path'])

    app.run(host='0.0.0.0', port=5005)
