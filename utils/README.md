# Mask Detection

A real time mask detection classifier developed to detect and classify faces with or without mask. The performance of the model is enhanced using OpenVino toolkit from Intel which enables faster CPU inference that lets users run AI modules such as this with minimal hardware. The initial model was trained using tensorflow-keras with the ResNet50 architecture as the base layers.  

## Pre-requisites
1. OpenVINO Toolkit 
2. imutils
3. OpenCV

## Running the inference
1. Set the path to video input in the file 'detect.py'
2. Run using the command 'python3 detect.py'

## Models directory
Contains openvino models for both face and mask detection, Also contains tensorflow .h5 and .pb converted mask detection model.

## To note:
In the file 'detect.py' the code to run the interface using tensorflow model exists. 