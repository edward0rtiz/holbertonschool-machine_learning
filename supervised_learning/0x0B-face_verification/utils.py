#!/usr/bin/env python3
"""
Face verification utils
"""

import os
import numpy as np
import cv2
import glob
import csv


def load_images(images_path, as_array=True):
    """
    Function to load images
    Args:
        images_path: the path to a directory from which to load images
        as_array: boolean indicating whether the images should be loaded
        as one numpy.ndarray
    Returns: images, filenames
             images: is either a list/numpy.ndarray of all images
             filenames: is a list of the filenames associated with each
                       image in images
    """
    images = []
    filenames = []

    image_path = glob.glob(images_path + "/*")
    image_path.sort()
    for path in image_path:
        img_name = path.split("/")[-1]
        filenames.append(img_name)

    for path in image_path:
        image = cv2.imread(path)
        new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(new_img)

    if as_array is True:
        images = np.array(images)

    return images, filenames


def load_csv(csv_path, params={}):
    """
    Load CSV content
    Args:
        csv_path: the path to the csv to load
        params: the parameters to load the csv with
    Returns: list of lists representing the contents
            found in csv_path
    """

    list_csv = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file, params)
        for row in csv_reader:
            list_csv.append(row)
    return list_csv

