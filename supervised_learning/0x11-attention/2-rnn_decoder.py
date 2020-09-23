# !/usr/bin/env python3
""" Machine Translation model with RNN's """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """

    """
    def __init__(self, vocab, embedding, units, batch):
        """

        Args:
            vocab:
            embedding:
            units:
            batch:
        """
        super(RNNDecoder, self).__init__()
        # self.batch = batch
        # self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform',
                                       return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(vocab)


    def call(self, x, s_prev, hidden_states):
        """

        Args:
            x:
            initial:

        Returns:
        """
        embedding = self.embedding(x)
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)
        inputs = tf.concat([embedding, context], -1)
        decode_outputs, state = self.gru(inputs, initial_state=hidden_states[:, 1])
        y = tf.reshape((decode_outputs), [-1, decode_outputs.shape[2]])
        y = self.F(y)

        return y, state
