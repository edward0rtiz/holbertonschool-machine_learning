#!/usr/bin/env python3
""" Multihead attention"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """multi head attention"""

    def __init__(self, dm, h):
        """
        Init method
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm

        assert dm % self.h == 0

        self.depth = dm // self.h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size,
        num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        call method
        """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)  # (batch_size, seq_len, d_model)
        k = self.Wk(K)  # (batch_size, seq_len, d_model)
        v = self.Wv(V)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size,
                                       -1,
                                       self.dm))

        output = self.linear(concat_attention)
        return output, attention_weights