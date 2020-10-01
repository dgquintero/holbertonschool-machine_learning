#!/usr/bin/env python3
"""Initialize YOLOv3 class"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Class Yolov3 to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor
        Arguments:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names used
                for the Darknet model, listed in order of index, can be found
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: containing all of the anchor boxes
        Returns:
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """

        Argumnets:
            outputs: predictions from the Darknet model for a single image
            image_size. ndarray containing the image original size
        Returns: (boxes, box_confidences, box_class_probs)
        """
        boxes = [pred[:, :, :, 0:4]for pred in outputs]
        for 
