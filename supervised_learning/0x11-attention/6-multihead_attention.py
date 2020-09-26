#!/usr/bin/env python3
""" Multihead attention"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """multihead attention"""

    def __init__(self, dm, h):
        """ init method
        Args:
        dm: integer representing the dimensionality of the model
        h: integer representing the number of heads
        dm: is divisible by h
        Public instance attributes:
        h: the number of heads
        dm: the dimensionality of the model
        depth: the depth of each attention head
        Wq: Dense layer with dm units, for generate the query matrix
        Wk: Dense layer with dm units, for generate the key matrix
        Wv: Dense layer with dm units, for generate the value matrix
        linear: Dense layer with dm units, for generate the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm / self.h)
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def call(self, Q, K, V, mask):
        """ Call transformers model
        Args:
        Q: tensor of shape (batch, seq_len_q, dk) containing the input to
           generate the query matrix
        K: tensor of shape (batch, seq_len_v, dk) containing the input to
           generate the key matrix
        V: tensor of shape (batch, seq_len_v, dv) containing the input to
           generate the value matrix
        mask: is always None
        Return: output, weights
        output: tensor with its last two dimensions as (..., seq_len_q, dm)
                containing the scaled dot product attention
        weights: tensor with its last three dimensions as
                (..., h, seq_len_q, seq_len_v) containing the attention weights
        """
        batch = Q.shape[0]

        V_linear = self.Wv(V)
        K_linear = self.Wk(K)
        Q_linear = self.Wq(Q)

        V = tf.reshape(V_linear, [batch, -1, self.h, self.depth])
        K = tf.reshape(K_linear, [batch, -1, self.h, self.depth])
        Q = tf.reshape(Q_linear, [batch, -1, self.h, self.depth])

        V = tf.transpose(V, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])

        output, weights = sdp_attention(Q, K, V, mask)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        outputs = self.linear(output)

        return outputs, weights
