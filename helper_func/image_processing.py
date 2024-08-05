""" Image processing

this file contains classes and functions for data processing

environment: Python 3.11 under Anaconda 23.7.4

"""

__author__ = "Ignatius S. Condro Atmawan"
__contact__ = "saptocondro@gmail.com"
__copyright__ = "none"
__date__ = "2024/08/05"

import numpy as np
import cv2


def load_image_pixels(image, shape):
    """
    load image pixels

    :param image: image from Open CV
    :param shape: contains width and height
    :return:
    """
    width, height = shape
    image = cv2.resize(image, (width, height))
    image = image.astype("float32")
    image /= 255.0
    image = np.expand_dims(image, 0)
    return image


def draw_bounding_boxes(image, boxes, scores, classes, class_names):
    """
    Draw bounding boxes of a person

    :param image: image from open CV
    :param boxes: boxes
    :param scores: classification scores
    :param classes: yolo classes
    :param class_names: coco class name
    :return:
    """
    for box, score, cls in zip(boxes, scores, classes):
        if cls == 0:  # Filter for class "person"
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{class_names[cls]}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

# end of file
