#!/usr/bin/env python3
""" sdp attention """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """

    Args:
        Q: tensor with its last two dimensions as (..., seq_len_q, dk)
           containing the query matrix
        K: tensor with its last two dimensions as (..., seq_len_v, dk)
           containing the key matrix
        V: tensor with its last two dimensions as (..., seq_len_v, dv)
           containing the value matrix
        mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
              containing the optional mask, or defaulted to None
        output, weights
    Returns:
    """

    q = tf.matmul(Q, K, transpose_b=True)
    # scale q
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_q = q / tf.math.sqrt(dk)

    if mask is not None:
        scaled_q += (mask * -1e9)

    weights = tf.nn.softmax(scaled_q, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights
