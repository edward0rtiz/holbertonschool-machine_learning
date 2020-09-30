#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention

np.random.seed(0)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 10, 256)).astype('float32'))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype('float32'))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 512)).astype('float32'))
output, weights = sdp_attention(Q, K, V)
O = tf.keras.backend.eval(output)
W = tf.keras.backend.eval(weights)
print(O.shape, np.array2string(O, precision=5))
print(W.shape, np.array2string(W, precision=5))