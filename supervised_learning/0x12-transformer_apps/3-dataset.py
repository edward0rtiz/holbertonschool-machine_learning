#!/usr/bin/env python3
"""Class dataset"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """Class Dataset"""

    def __init__(self, batch_size, max_len):
        """Class constructor"""

        def filter_max_length(x, y, max_length=max_len):
            """
            filter method
            """
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

        # train
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(filter_max_length)
        # cache the dataset to memory
        self.data_train = self.data_train.cache()

        shuff = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(shuff)
        pad_shape = ([None], [None])
        self.data_train = self.data_train.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)
        aux = tf.data.experimental.AUTOTUNE
        self.data_train = self.data_train.prefetch(aux)

        # valid
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(batch_size,
                                                       padded_shapes=pad_shape)

    def tokenize_dataset(self, data):
        """tokenize data """

        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encoding """

        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(self, pt, en):
        """ tf_encode"""
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
