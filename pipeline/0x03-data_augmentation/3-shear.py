#!/usr/bin/env python3
""" Shear image """
import tensorflow as tf


def shear_image(image, intensity):
    """shear image"""
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared_img = tf.keras.preprocessing.image.random_shear(img, intensity,
                                                            row_axis=0,
                                                            col_axis=1,
                                                            channel_axis=2
                                                            )
    image_out = tf.keras.preprocessing.image.array_to_img(sheared_img)
    return image_out
