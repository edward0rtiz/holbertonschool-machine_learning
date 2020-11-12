#!/usr/bin/env python3
""" flip_image"""
import tensorflow as tf


def flip_image(image):
    """flip image"""
    flip = tf.image.flip_left_right(image)
    return flip
