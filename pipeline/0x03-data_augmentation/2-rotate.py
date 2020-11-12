#!/usr/bin/env python3
""" rotate_image"""
import tensorflow as tf


def rotate_image(image):
    """rotates image
    """
    img_90 = tf.image.rot90(image, k=1)
    return img_90
