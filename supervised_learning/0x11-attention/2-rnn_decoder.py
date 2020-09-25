#!/usr/bin/env python3
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
        batch, units = s_prev.shape
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        embeddings = self.embedding(x)
        concat_input = tf.concat([tf.expand_dims(context, 1),
                                  embeddings],
                                 axis=-1)
        # passing the concatenated vector to the GRU
        outputs, hidden = self.gru(concat_input)
        # output shape == (batch_size * 1, hidden_size)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)
        return y, hidden
