#!/usr/bin/env python3
"""Script to initialize YOLOv3"""

import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


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
            image_size: numpy.ndarray containing the image’s original size
                        [image_height, image_width]
        Returns: (boxes, box_confidences, box_class_probs)
        """
        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        for ipred, pred in enumerate(boxes):
            for grid_h in range(pred.shape[0]):
                for grid_w in range(pred.shape[1]):
                    bx = ((self.sigmoid(pred[grid_h,
                                        grid_w, :,
                                        0]) + grid_w) / pred.shape[1])
                    by = ((self.sigmoid(pred[grid_h,
                                        grid_w, :,
                                        1]) + grid_h) / pred.shape[0])
                    anchor_tensor = self.anchors[ipred].astype(float)
                    anchor_tensor[:, 0] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               2]) / self.model.input.shape[1].value  # bw
                    anchor_tensor[:, 1] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               3]) / self.model.input.shape[2].value  # bh

                    pred[grid_h, grid_w, :, 0] = \
                        (bx - (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]  # x1
                    pred[grid_h, grid_w, :, 1] = \
                        (by - (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]  # y1
                    pred[grid_h, grid_w, :, 2] = \
                        (bx + (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]  # x2
                    pred[grid_h, grid_w, :, 3] = \
                        (by + (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]  # y2
        # box confidence

        box_confidences = [self.sigmoid(pred[:, :, :,
                                        4:5]) for pred in outputs]

        # box class probs
        box_class_probs = [self.sigmoid(pred[:, :, :,
                                        5:]) for pred in outputs]
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Function that filter boxes
        Args:
            boxes: List of numpy.ndarrays of shape (grid_height, grid_width,
                   anchor_boxes, 4) containing the processed boundary boxes
                   for each output, respectively
            box_confidences: list of numpy.ndarrays of shape (grid_height,
                             grid_width, anchor_boxes, 1) containing the
                             processed box confidences for each output,
                             respectively
            box_class_probs: list of numpy.ndarrays of shape (grid_height,
                             grid_width, anchor_boxes, classes) containing
                             the processed box class probabilities for each
                             output, respectively
        Returns: Tuple of (filtered_boxes, box_classes, box_scores)
        """
        box_score = []
        bc = box_confidences
        bcp = box_class_probs

        for box_conf, box_probs in zip(bc, bcp):
            score = (box_conf * box_probs)
            box_score.append(score)
        # Finding the index of the class with maximum box score
        box_classes = [s.argmax(axis=-1) for s in box_score]
        box_class_l = [b.reshape(-1) for b in box_classes]
        box_classes = np.concatenate(box_class_l)

        # Getting the corresponding box score
        box_class_scores = [s.max(axis=-1) for s in box_score]
        b_scores_l = [b.reshape(-1) for b in box_class_scores]
        box_class_scores = np.concatenate(b_scores_l)

        # Filter mask (pc >= threshold)
        mask = np.where(box_class_scores >= self.class_t)

        # Filtered all unbounding boxes
        boxes_all = [b.reshape(-1, 4) for b in boxes]
        boxes_all = np.concatenate(boxes_all)

        # Applying the mask to scores, boxes and classes
        scores = box_class_scores[mask]
        boxes = boxes_all[mask]
        classes = box_classes[mask]

        return boxes, classes, scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Max suppression function
        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4) containing
                            all of the filtered bounding boxes:
            box_classes: numpy.ndarray of shape (?,) containing the
                         class number for the class that filtered_boxes
                         predicts, respectively
            box_scores: numpy.ndarray of shape (?) containing the box scores
                        for each box in filtered_boxes, respectively
        Returns: box_predictions, predicted_box_classes, predicted_box_scores
        """
        f = []
        c = []
        s = []

        for i in(np.unique(box_classes)):

            idx = np.where(box_classes == i)
            filters = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]
            keep = self.nms(filters, self.nms_t, scores)

            filters = filters[keep]
            scores = scores[keep]
            classes = classes[keep]

            f.append(filters)
            c.append(classes)
            s.append(scores)

        filtered_boxes = np.concatenate(f, axis=0)
        box_scores = np.concatenate(c, axis=0)
        box_classes = np.concatenate(s, axis=0)

        return filtered_boxes, box_scores, box_classes

    def nms(self, bc, thresh, scores):
        """
        Function that computes the index
        Args:
            bc: Box coordinates
            thresh: Threeshold
            scores: scores for each box indexed and sorted
        Returns: Sorted index score for non max supression
        """
        x1 = bc[:, 0]
        y1 = bc[:, 1]
        x2 = bc[:, 2]
        y2 = bc[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def load_images(folder_path):
        """
        Function to load images using cv2
        Args:
            folder_path: String representing the path to the folder
                         holding all the images to load
        Returns: tuple of (images, image_paths)
        """
        images = []
        images_paths = glob.glob(folder_path + '/*', recursive=False)
        for i in images_paths:
            image = cv2.imread(i)
            images.append(image)
        return images, images_paths

    def preprocess_images(self, images):
        """
        Function that preprocess the images
        Args:
            images:
        Returns:
        """
        pimages = []
        shapes = []
        input_h = self.model.input.shape[2].value
        input_w = self.model.input.shape[1].value
        for i in images:
            img_shape = i.shape[0], i.shape[1]
            shapes.append(img_shape)
            image = cv2.resize(i, (input_w, input_h),
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255
            pimages.append(image)
        pimages = np.array(pimages)
        image_shapes = np.array(shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores,
                   file_name):
        """
        Function to draw boxes
        Args:
            image: np.ndarray containing an unprocessed image
            boxes: np.ndarray containing the boundary boxes
                   for the image
            box_classes: np.array  containing the class indices for
                         each box
            box_scores: np.array containing the box scores for each box
            file_name: file path where the original image is stored
        Returns: Image with all boundary boxes, class names, and box scores
        """

        for i, box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])
            start_point = int(box[0]), int(box[1])
            end_point = int(box[2]), int(box[3])
            scores = "{:.2f}".format(box_scores[i])
            label = (self.class_names[box_classes[i]] + " " + scores)
            oorg = (x1, y1 - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            text_color = (0, 0, 255)
            thick = 1
            line_Type = cv2.LINE_AA
            image = cv2.rectangle(image, start_point, end_point,
                                  (255, 0, 0), thickness=2)
            print(image)
            image = cv2.putText(image, label, oorg, font, scale, text_color,
                                thick, line_Type, bottomLeftOrigin=False)
        cv2.imshow(file_name, image)

        k = cv2.waitKey(0)
        if k == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            os.chdir('detections')
            cv2.imwrite(file_name, image)
        cv2.destroyAllWindows()
