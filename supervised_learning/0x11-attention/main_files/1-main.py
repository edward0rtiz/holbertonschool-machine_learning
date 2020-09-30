#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

np.random.seed(0)
tf.set_random_seed(0)
attention = SelfAttention(256)
print(type(attention.W), attention.W.units)
print(type(attention.U), attention.U.units)
print(type(attention.V), attention.V.units)

with open('1-test', 'w+') as f:
    s_prev = tf.convert_to_tensor(np.random.uniform(size=(32, 256)).astype('float32'))
    hidden_states = tf.convert_to_tensor(np.random.uniform(size=(32, 10, 256)).astype('float32'))
    context, weights = attention(s_prev, hidden_states)
    C = tf.keras.backend.eval(context)
    W = tf.keras.backend.eval(weights)
    f.write(str(C.shape) + '\n' + np.array2string(C, precision=5) + '\n')
    f.write(str(W.shape) + '\n' + np.array2string(W, precision=5) + '\n')