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


def save_images(path, images, filenames):
    """
    Save images in specific path
    Args:
        path: path to the directory in which the
              images should be saved
        images: list/numpy.ndarray of images to save
        filenames: list of filenames of the images to save
    Returns: True on success and False on failure
    """
    if not os.path.exists(path):
        return False
    else:
        for i in range(len(images)):
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path, filenames[i]), img)
        return True


def generate_triplets(images, filenames, triplet_names):
    """
    Fucntion to create triplets
    Args:
        images: numpy.ndarray of shape (n, h, w, 3) containing
        the various images in the dataset
        filenames: list of length n containing the corresponding
                    filenames for images
        triplet_names: list of lists where each sublist contains
                       the filenames of an anchor, positive, and
                       negative image, respectively
    Returns: list [A, P, N]
    """

    anchor, positive, negative = [], [], []
    _, h, w, c = images.shape

    new_file = [filenames[i].split('.')[0] for i in range(len(filenames))]

    for i in range(len(triplet_names)):
        a, p, n = triplet_names[i]

        if a in new_file:
            if p in new_file:
                if n in new_file:
                    idx_a = new_file.index(a)
                    idx_p = new_file.index(p)
                    idx_n = new_file.index(n)

                    img_a = images[idx_a]
                    img_p = images[idx_p]
                    img_n = images[idx_n]

                    anchor.append(img_a)
                    positive.append(img_p)
                    negative.append(img_n)

    anchor = [i.reshape(1, h, w, c) for i in anchor]
    anchor = np.concatenate(anchor)
    positive = [i.reshape(1, h, w, c) for i in positive]
    positive = np.concatenate(positive)
    negative = [i.reshape(1, h, w, c) for i in negative]
    negative = np.concatenate(negative)

    return [anchor, positive, negative]
