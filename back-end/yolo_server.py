#!/usr/bin/python3

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import json
from flask import Flask , session, redirect, request, Response
from pprint import pprint
from datetime import datetime
import threading as t

app = Flask(__name__)
app.secret_key = "secret_key"



@app.route("/processframe", methods=['GET'])
def processFrame():
    before = datetime.now()

    r = request.get_data()
    if r is None:
        print("Empty request")
        return "Empty request", 400

    shape = request.headers.get('shape')
    frame = np.frombuffer(r, dtype=dType)
    frame = np.reshape(frame, json.loads(shape)) 
    
    try:
        cv_lock.acquire()
        
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (blobWidth, blobHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        retData = postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        inferenceTime = t * 1000.0 / cv.getTickFrequency()

        cv_lock.release()
    except Exception as e:
        raise(e)
        print("Malformed input")
        return "Malformed input", 400
    
    resp = Response(json.dumps(retData))
    resp.headers['inferenceTime'] = inferenceTime

    after = datetime.now()

    resp.headers['serverTime'] = (after - before).total_seconds() * 1000
    return resp, 200

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    retData = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        retData.append({'class': int(classIds[i]), 'confidence': confidences[i], 'left': left, 'top': top, 'leftWidth': left + width, 'topHeight': top + height})
    
    return retData


if __name__ == "__main__":
    confThreshold = 0.5 #Confidence threshold
    nmsThreshold = 0.4 #Non-maximum suppression threshold

    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"
 
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    dType = 'uint8'

    blobHeight = 416
    blobWidth = 416

    cv_lock = t.Lock()

    app.run(host='0.0.0.0', debug=True)