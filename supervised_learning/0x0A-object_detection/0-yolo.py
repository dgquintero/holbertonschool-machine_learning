#!/usr/bin/env python3
"""Initialize class YOLOv3"""

import tensorflow.keras as K


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
