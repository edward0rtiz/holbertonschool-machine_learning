#!/usr/bin/env python3
""" Main model using RNN-LSTM single-step"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class WindowGenerator:
    """ Class window genenrator """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """
        Initialize method
        Args:
            input_width:
            label_width:
            shift:
            train_df:
            val_df:
            test_df:
            label_columns:
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = \
            np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = \
            np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """
        Split window
        Args:
            features:
        Returns:
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        Make data set
        Args:
            data:

        Returns:

        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        """
        train
        Returns:

        """
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """
        val
        Returns:

        """
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """
        test
        Returns:
        """
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """
        Plot function
        Args:
            model:
            plot_col:
            max_subplots:
        Returns:
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,
                                                                 None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker='X', edgecolors='k',
                            label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    class Baseline(tf.keras.Model):
        """
        Baseline class
        """
        def __init__(self, label_index=None):
            """
            init
            Args:
                label_index:
            """
            super().__init__()
            self.label_index = label_index

        def call(self, inputs):
            """
            call
            Args:
                inputs:

            Returns:

            """
            if self.label_index is None:
                return inputs
            result = inputs[:, :, self.label_index]
            return result[:, :, tf.newaxis]

    def compile_and_fit(model, window, patience=2):
        """
        compile and fit
        Args:
            window:
            patience:
        Returns:
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history

    def build_model():
        """
        build_model
        Returns:
        """
        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(16, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])
        lstm_model.summary()

        return lstm_model
