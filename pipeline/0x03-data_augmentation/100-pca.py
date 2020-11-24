#!/usr/bin/env python3
"""
PCA alex-net by:
Thanks to
https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
"""

import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    pca color augmentation based on alex-net paper
    Args:
        image: 3D tf.Tensor containing the image to change
        alphas: tuple of length 3 containing the amount
                that each channel should change
    Returns: Returns the augmented image
    """
    # getting the original image from tf
    original_image = tf.keras.preprocessing.image.img_to_array(image)
    cp_original = original_image.astype(float).copy()


    # flatten image to columns of RGB
    original_image = original_image / 255.0
    img_rs = original_image.reshape(-1, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)


    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))

    # broad cast to speed things up
    m2[:, 0] = alphas * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    # RGB
    for idx in range(3):
        cp_original[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # cp_original /= 255.0
    cp_original = np.clip(cp_original, 0.0, 255.0)

    # cp_original *= 255
    cp_original = cp_original.astype(np.uint8)

    # about 100x faster after vectorizing the numpy, it will be even faster later
    # since currently it's working on full size images and not small, square
    # images that will be fed in later as part of the post processing before being
    # sent into the model
    return cp_original
