# Mask Detection

A real time mask detection classifier developed to detect and classify faces with or without mask. The performance of the face detection model is enhanced using OpenVino toolkit from Intel which enables faster CPU inference. The initial mask detection model was trained using tensorflow and keras with the ResNet50 architecture.

![img](https://github.com/vyzboy92/mask-detection/blob/master/utils/demo.jpg)

## Pre-requisites
1. OpenVINO Toolkit 
2. imutils
3. OpenCV
4. Kers
5. Tensorflow

## Running the inference
1. Set the path to video input in the file 'detect.py'
2. Run using the command ``` python3 detect.py ```
