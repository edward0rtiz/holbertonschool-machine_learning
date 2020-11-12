#!/usr/bin/env python3
""" hue changing image"""
import tensorflow as tf


def change_hue(image, delta):
    """changes the hue of an image"""
    img = tf.image.adjust_hue(image, delta)
    return img
