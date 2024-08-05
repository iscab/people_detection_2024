""" People Detection

Procedure of data processing and classifier inferencing:
1. load video
2. prepare classifier model, for example YOLO
3. calculate number of object "person"
4. calculate time or frame with the label "person"

data sources:  https://www.pexels.com/search/videos/crowd/

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
# print(class_names)
# print(class_names[0])

# Set YOLOv3 anchors and masks
yolo_anchors = np.array([
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)]
], np.float32) / 416

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

    # Run YOLOv3 model to get predictions
    yhat = yolo_model.predict(input_image)

    # Post-process YOLOv3 predictions
    boxes, scores, classes = [], [], []
    print(len(yhat))
    for i in range(len(yhat)):
        # print(yhat[i].shape)
        # print(yhat[i][0, 1, 1, :4])
        # exit()
        grid123, grid_h, grid_w, _ = yhat[i].shape
        for row in range(grid_h):
            for col in range(grid_w):
                box = yhat[i][0, row, col, :4]
                confidence = yhat[i][0, row, col, 4]
                class_probs = yhat[i][0, row, col, 5:]
                # print(class_probs)
                """for anchor in range(3):
                    box = yhat[i][row, col, anchor, :4]
                    confidence = yhat[i][row, col, anchor, 4]
                    class_probs = yhat[i][row, col, anchor, 5:]

                    class_id = np.argmax(class_probs)
                    class_score = class_probs[class_id]

                    if confidence * class_score > 0.5:
                        center_x, center_y, width, height = box
                        center_x = (center_x + col) / grid_w
                        center_y = (center_y + row) / grid_h
                        width = width * yolo_anchors[i][anchor][0] / 416
                        height = height * yolo_anchors[i][anchor][1] / 416

                        box = [center_x, center_y, width, height]
                        boxes.append(box)
                        scores.append(confidence * class_score)
                        classes.append(class_id)

    # Draw bounding boxes on the frame
    annotated_frame = hfun_img.draw_bounding_boxes(frame, boxes, scores, classes, class_names)

    # Display the frame
    cv2.imshow("People Detection", annotated_frame)
    out.write(annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break"""

# clear
cap.release()
out.release()
cv2.destroyAllWindows()


print("End of script")
# end of file
