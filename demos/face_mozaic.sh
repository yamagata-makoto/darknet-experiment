#!/bin/bash

if [ $# -ne 1 ]; then
  echo "run_face <jpg>"
  exit 1
fi
if [ ! -e "yolov3-face.weights" ]; then
  wget https://www.dropbox.com/s/0np6vyaliu6mv65/yolov3-face.weights
fi
if [ ! -e "../darknet" ]; then
  echo "darknet not found..."
  exit 1
fi
../darknet facecv detect yolov3-face.cfg yolov3-face.weights $1

