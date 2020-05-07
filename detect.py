#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import datetime as dt
import dlib
import face_recognition
import imutils
import logging as log
import numpy as np
import os
import requests
import subprocess
import sys
import time
from imutils import face_utils
from imutils.video import WebcamVideoStream
from multiprocessing import Process, Queue
from openvino.inference_engine import IENetwork, IEPlugin
from tensorflow.keras import models

# importing the mask detection model
model_mask = models.load_model("utils/mask_detection.model")

# declaring the classes
classes = ["masked", "unmasked"]


# container function to initialise OpenVINO models
def init_model(xml, bins):
    model_xml = xml
    model_bin = bins
    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU')
    plugin.add_cpu_extension(
        'utils/libcpu_extension_sse4.so')
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_nets = plugin.load(network=net, num_requests=2)
    n, c, h, w = net.inputs[input_blob].shape
    del net
    return exec_nets, n, c, w, h, input_blob, out_blob, plugin


def mask_detection(image):
    output = image.copy()
    output = imutils.resize(output, width=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32")
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    image -= mean
    preds = model_mask.predict(np.expand_dims(image, axis=0))[0]
    i = np.argmax(preds)
    label = classes[i]
    return label


def main():
    # paths to models
    face_xml = "utils/face-detection-adas-0001.xml"
    face_bin = "utils/face-detection-adas-0001.bin"
    face_detection_net, n_f, c_f, w_f, h_f, input_blob_f, out_blob_f, plugin_f = init_model(face_xml, face_bin)

    fvs = WebcamVideoStream(src=0).start()
    time.sleep(0.5)

    # Initialize some variables
    frame_count = 0
    cur_request_id_f = 0
    next_request_id_f = 1

    while True:
        # Grab a single frame of video
        frame = fvs.read()
        initial_h, initial_w = frame.shape[:2]
        if frame is None:
            break

        # Find all the faces and face encodings in the current frame of video
        face_locations = []
        in_frame = cv2.resize(frame, (w_f, h_f))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n_f, c_f, h_f, w_f))
        face_detection_net.start_async(request_id=cur_request_id_f, inputs={input_blob_f: in_frame})
        if face_detection_net.requests[cur_request_id_f].wait(-1) == 0:
            face_detection_res = face_detection_net.requests[cur_request_id_f].outputs[out_blob_f]
            for face_loc in face_detection_res[0][0]:
                if face_loc[2] > 0.5:
                    xmin = abs(int(face_loc[3] * initial_w))
                    ymin = abs(int(face_loc[4] * initial_h))
                    xmax = abs(int(face_loc[5] * initial_w))
                    ymax = abs(int(face_loc[6] * initial_h))
                    face_locations.append((xmin, ymin, xmax, ymax))

        for (left, top, right, bottom) in face_locations:
            crop_img = frame[top:bottom, left:right]
            label = mask_detection(crop_img)
            frame = cv2.rectangle(frame, (left, top), (right, bottom), (65, 65, 65), 2)
            cv2.putText(frame, label, (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        cv2.imshow('Video', frame)
        frame_count += 1

        cur_request_id_f, next_request_id_f = next_request_id_f, cur_request_id_f

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    fvs.stop()
    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    main()
