#!/usr/bin/enoutput python3
""" sdp attention """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """

    Args:
        Q:
        K:
        V:
        mask:

    Returns:
    """
    q = tf.matmul(Q, K, transpose_b=True)
    k = tf.cast((tf.math.square(K.shape[-1])), tf.float32)
    qk_scale = tf.math.divide(q, k)
    if mask is not None:
        mask_mul = tf.math.multiply(mask, -1e9)
        qk_scale += mask_mul
    weights = tf.nn.softmax(qk_scale, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
