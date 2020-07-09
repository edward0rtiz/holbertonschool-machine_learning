#!/usr/bin/env python3
"""
Triplet_loss
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Layer


class TripletLoss(Layer):
    """
    Triplet_loss class
    """
    def __init__(self, alpha, **kwargs):
        super().__init__(*kwargs)
        self.alpha = alpha
        self._dynamic = True
        self._eager_losses = True  # OK
        self.__layers = True
        self._in_call = False
        self._metrics = True
        self._metrics_tensors = True
        self._mixed_precision_policy = True
        self._obj_reference_counts_dict = True
        self._self_setattr_tracking = True


    def triplet_loss(self, inputs):
        """
        Triptel_losss function
        Args:
            inputs: list containing the anchor,
                    positive and negative output
        Returns: Tensor containing the triplet loss values
        """
        anchor, positive, negative = inputs
        pos_dist = K.backend.sum(K.backend.square(anchor - positive), axis=-1)
        neg_dist = K.backend.sum(K.backend.square(anchor - negative), axis=-1)

        basic_loss = K.layers.Subtract()([pos_dist, neg_dist]) + self.alpha

        loss = K.backend.maximum(basic_loss, 0)
        return loss

    def call(self, inputs):
        """
        Call function
        Args:
            inputs: list containing the anchor, positive, and negative output
        Returns: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss