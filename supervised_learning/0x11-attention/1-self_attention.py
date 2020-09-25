#!/usr/bin/env python3
""" Machine Translation model with RNN's """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class Self attention
    """
    def __init__(self, units):
        """
        method initializer
        Args:
            units: Integer representing the number of hidden units in the
                   alignment model

            W - Dense layer with units units, to be applied to the previous
                decoder hidden state
            U - Dense layer with units units, to be applied to the
                encoder hidden states
            V - Dense layer with 1 units, to be applied to the tanh of
                the sum of the outputs of W and U
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        call method
        Args:
            s_prev: tensor of shape (batch, units) containing the previous
                    decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                           containing the outputs of the encoder
        Returns: context, weights
                 context: Tensor of shape (batch, units) that contains
                          the context vector for the decoder
                 weights: Tensor of shape (batch, input_seq_len, 1)
                          that contains the attention weights
        """
        u = self.U(hidden_states)
        w = tf.expand_dims(s_prev, axis=1)
        # score of hidden state
        s = self.V((tf.nn.tanh(w + u)))
        attention_w = tf.nn.softmax(s, axis=1)
        context = tf.reduce_sum((attention_w * hidden_states), axis=1)

        return context, attention_w
