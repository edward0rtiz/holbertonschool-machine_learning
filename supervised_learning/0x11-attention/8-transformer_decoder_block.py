#!/usr/bin/env python3
""" Tranformer decoder block"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ class DecoderBlock """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Init method
        Args:
            dm: the dimensionality of the model
            h: the number of heads
            hidden: the number of hidden units in the fully connected layer
            drop_rate: the dropout rate
        """
        super().__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        call public instance method
        Args:
            x: Tensor of shape (batch, target_seq_len, dm)containing the i
                nput to the decoder block
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                            containing the output of the encoder
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to the first multi head
                             attention layer
            padding_mask: mask to be applied to the second multi head
                          attention layer
        Returns: tensor of shape (batch, target_seq_len, dm)
                 containing the block’s output
        """
        attention, attention_block = self.mha1(x, x, x, look_ahead_mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(attention + x)
        attention2, attn_weights_block2 = self.mha2(out1,
                                                    encoder_output,
                                                    encoder_output,
                                                    padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(attention2 + out1)
        hidden_output = self.dense_hidden(out2)
        output_output = self.dense_output(hidden_output)
        ffn_output = self.dropout3(output_output, training=training)
        output = self.layernorm3(ffn_output + out2)

        return output
