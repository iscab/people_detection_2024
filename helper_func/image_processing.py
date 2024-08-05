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


class BoundBox:
    """  Bounding Box  """
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        """"""
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        """"""
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        """"""
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    """
    Sigmoid function

    :param x: number
    :return: sigmoid function result
    """
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    """"""
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)

    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    """"""
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    """"""
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3


def bbox_iou(box1, box2):
    """
    Intersection over union

    :param box1:
    :param box2:
    :return:
    """
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union









# end of file
