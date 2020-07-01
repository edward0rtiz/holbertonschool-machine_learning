#!/usr/bin/env python3
"""Script to initialize YOLOv3"""

import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    Class YOLOv3
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Method init for Yolov3
        Args:
            model_path: path to where a Darknet nperas model is stored
            classes_path: path to where the list of class names used for
                          the Darknet model, listed in order of index,
                          can be found
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        # Load model
        self.model = K.models.load_model(model_path)
        # Load classes
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """ sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Method containing the predictions from the darknet_model
        Args:
            outputs: list of numpy.ndarrays containing the predictions from
                     the Darknet model for a single image:
                        (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size:

        Returns: (boxes, box_confidences, box_class_probs)
        """

        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        for ipred, pred in enumerate(boxes):
            for grid_w in range(pred.shape[0]):
                for grid_h in range(pred.shape[1]):
                    cx = ((self.sigmoid(pred[grid_w, grid_h, :, 0]) + grid_h) / pred.shape[1] * image_size[1])
                    cy = ((self.sigmoid(pred[grid_w, grid_h, :, 1]) + grid_w) / pred.shape[0] * image_size[0])
                    anchor_tensor = self.anchors[ipred].astype(float)
                    th = image_size[1] / self.model.input.shape[1].value
                    tw = image_size[0] / self.model.input.shape[2].value
                    anchor_tensor[:, 0] *= self.sigmoid(pred[grid_w, grid_h, :, 2]) / 2 * th
                    anchor_tensor[:, 1] *= self.sigmoid(pred[grid_w, grid_h, :, 3]) / 2 * tw
                    pred[grid_w, grid_h, :, 0] = cx - anchor_tensor[:, 0]  # x1
                    pred[grid_w, grid_h, :, 1] = cy - anchor_tensor[:, 1]  # y1
                    pred[grid_w, grid_h, :, 2] = cx + anchor_tensor[:, 0]  # x2
                    pred[grid_w, grid_h, :, 3] = cy + anchor_tensor[:, 1]  # y2
        # box confidence

        box_confidences = [self.sigmoid(pred[:, :, :, 4:5]) for pred in outputs]

        # box class probs
        box_class_probs = [self.sigmoid(pred[:, :, :, 5:]) for pred in outputs]
        return boxes, box_confidences, box_class_probs
