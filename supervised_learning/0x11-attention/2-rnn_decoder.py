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
            vocab: integer representing the size of the output vocabulary
            embedding: integer representing the dimensionality of the
                       embedding vector
            units: integer representing the number of hidden units in
                   the RNN cell
            batch:  integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        # self.batch = batch
        # self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Call function of the encoder
        Args:
            x: tensor of shape (batch, 1) containing the previous word in the
               target sequence as an index of the target vocabulary
            s_prev: tensor of shape (batch, units) containing the
                    previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                           containing the outputs of the encoder
        Returns: y, s
                y: tensor of shape (batch, vocab) containing the output word
                   as a one hot vector in the target vocabulary
                s: tensor of shape (batch, units) containing the new decoder
                   hidden state
        """
        embedding = self.embedding(x)
        attention = SelfAttention(s_prev.shape[1])
        context, weights = attention(s_prev, hidden_states)
        context = tf.expand_dims(context, axis=1)
        inputs = tf.concat([embedding, context], -1)
        decode_outputs, h_state = self.gru(inputs,
                                           initial_state=hidden_states[:, 1])
        y = tf.reshape((decode_outputs), [-1, decode_outputs.shape[2]])
        y = self.F(y)

        return y, h_state
