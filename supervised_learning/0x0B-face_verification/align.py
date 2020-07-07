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


    def find_landmarks(self, image, detection):
        """
        Find landmarks
        Args:
            image: numpy.ndarray of an image from which to find
                   facial landmarks
            detection: dlib.rectangle containing the boundary
                       box of the face in the image
        Returns: numpy.ndarray of shape (p, 2)containing the
                 landmark points, or None on failure
        """
        try:
            shape = self.shape_predictor(image, detection)
            coord = np.zeros((shape.num_parts, 2), dtype="int")
            for i in range(0, shape.num_parts):
                coord[i] = (shape.part(i).x, shape.part(i).y)

            return coord
        except Exception:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        Function to align for face verification
        Args:
            image: numpy.ndarray of rank 3 containing the image to be
                   aligned
            landmark_indices: numpy.ndarray of shape (3,) containing
                            the indices of the three landmark points
                            that should be used for the affine
                            transformation
            anchor_points: numpy.ndarray of shape (3, 2) containing
                           the destination points for the affine
                           transformation, scaled to the range [0, 1]
            size: the desired size of the aligned image
        Returns: numpy.ndarray of shape (size, size, 3) containing the
                 aligned image, or None if no face is detected
        """
        img = self.detect(image)
        landmark = self.find_landmarks(image, img)
        srcTri = landmark[landmark_indices]
        srcTri = srcTri.astype(np.float32)
        anchor = anchor_points * size
        warp_mat = cv2.getAffineTransform(srcTri, anchor)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

        return warp_dst
