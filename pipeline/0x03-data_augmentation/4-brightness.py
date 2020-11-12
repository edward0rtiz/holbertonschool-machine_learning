#!/usr/bin/env python3
""" brightness an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """brightness an image"""
    img = tf.image.adjust_brightness(image, max_delta)
    return img
