# !/usr/bin/env python3
""" Machine Translation model with RNN's """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """

    """
    def __init__(self, units):
        """

        Args:
            units:
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(units)

    def call(self, s_prev, hidden_states):
        """

        Args:
            s_prev:
            hidden_states:

        Returns:
        """
        w = self.W(s_prev)
        u = self.U(hidden_states)
        w = tf.expand_dims(w, axis=1)
        # he = hidden state
        he = self.V((tf.nn.tanh(w + u)))
        attention_w = tf.nn.softmax(he, axis=1)
        context = tf.reduce_sum((attention_w * hidden_states), axis=1)

        return context, attention_w
