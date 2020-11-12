#!/usr/bin/env python3
""" crop_image"""
import tensorflow as tf


def crop_image(image, size):
    """crop image"""
    img = tf.random_crop(image, size=size)
    return img