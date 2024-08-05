""" People Detection

Procedure of data processing and classifier inferencing:
1. load video
2. prepare classifier model, for example YOLO
3. calculate number of object "person"
4. calculate time or frame with the label "person"

data sources:  https://www.pexels.com/search/videos/crowd/

reference:
https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

environment: Python 3.11 and TensorFlow 2.15.0 under Anaconda 23.7.4

"""

__author__ = "Ignatius S. Condro Atmawan"
__contact__ = "saptocondro@gmail.com"
__copyright__ = "alfatraining"
__date__ = "2024/08/05"

import os
import datetime as dt

import numpy as np
import tensorflow as tf
import cv2

import helper_func.image_processing as hfun_img

print("Hello World!")

# prepare output folder
current_directory = os.getcwd()
output_directory = os.path.join(current_directory, "output")
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
# time stamp format for output files
standard_date_time_format = "_%Y_%m_%d_%H_%M_%S"

# classifier model, with YOLO
model_path = r"model_data\yolov3.h5"
yolo_model = tf.keras.models. load_model(model_path)

# Load YOLOv3 class labels
label_path = r"model_data\coco.names"
with open(label_path, 'r') as f:
    class_names = f.read().splitlines()


print("End of script")
# end of file
