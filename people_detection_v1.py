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

num_of_people_threshold = 4
time_range_in_minutes = 2

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

# Set YOLOv3 anchors
yolo_anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# Set the probability threshold for detected objects
class_threshold = 0.6


# Video as input
# video_path = r"sample_video\3699517-hd_1920_1080_30fps.mp4"
video_path = r"sample_video\3747854-uhd_3840_2160_24fps.mp4"
cap = cv2.VideoCapture(video_path)

# output file
output_file = r"output"
ticktime = dt.datetime.now()
file_name_stamp = ticktime.strftime(standard_date_time_format)
output_file += file_name_stamp + ".mp4"
output_file = os.path.join(output_directory, output_file)
# print(output_file)

# Video as output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Video processing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for YOLOv3
    input_image = hfun_img.load_image_pixels(frame, (416, 416))
    # print(input_image.shape)
    # break

    # Run YOLOv3 model to get predictions
    yhat = yolo_model.predict(input_image)

    # Post-process YOLOv3 predictions
    boxes = []
    for i in range(len(yhat)):
        # print(i, yhat[i].shape)
        boxes += hfun_img.decode_netout(yhat[i][0], yolo_anchors[i], class_threshold, 416, 416)
    # print(len(boxes))
    # break

    # suppress non-maximal boxes
    hfun_img.do_nms(boxes, 0.6)
    # print(len(boxes))
    # break

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = hfun_img.get_boxes(boxes, class_names, 0.5)
    # print(len(v_boxes))
    # break

    # summarize what we found and count the number of people
    num_of_people = 0
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])
        # calculate number of people
        if v_labels[i] == "person":
            num_of_people += 1
    print(f"number of people = {num_of_people}")

    # prepare alert if the number of people achieve threshold
    if num_of_people >= num_of_people_threshold:
        print(f"number of people >= {num_of_people_threshold}")
    # break




# clear
cap.release()
out.release()
cv2.destroyAllWindows()


print("End of script")
# end of file
