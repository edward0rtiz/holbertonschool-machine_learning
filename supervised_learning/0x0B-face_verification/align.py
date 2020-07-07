#!/usr/bin/env python3
"""
Face Align script
"""

import numpy as np
import dlib
import cv2



class FaceAlign:
    """
    Class FaceAlign
    Args:
        self:
        shape_predictor_path: path to the dlib shape
                            predictor modelSets the public
                            instance attributes
        Attributes:
            detector: contains dlib‘s default face detector
            shape_predictor: contains the dlib.shape_predictor
    Returns: type object
    """
    def __init__(self, shape_predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)


    def detect(self, image):
        """
        Detecs a face in an image
        Args:
            image: numpy.ndarray of rank 3 containing an image from which
                   to detect a face
        Returns: dlib.rectangle containing the boundary box for the face
                in the image, or None on failure
        """
        try:
            faces = self.detector(image, 1)
            max_area = 0
            rectangle = (dlib.rectangle(0, 0, image.shape[1],
                                        image.shape[0]))

            if len(faces) >= 1:
                for face in faces:
                    if face.area() > max_area:
                        max_area = face.area()
                        rectangle = face
            else:
                rectangle

            return rectangle
        except Exception:
            return None
