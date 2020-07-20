#!/usr/bin/env python3
"""
Object verification model
"""
import tensorflow as tf
import tensorflow.keras as K
from triplet_loss import TripletLoss
import numpy as np

#tf.enable_eager_execution()


class TrainModel():
    def __init__(self, model_path, alpha):
        """
        Initialize model
        Args:
            model_path: path to the base face verification
                        embedding model
            alpha: alpha to use for the triplet loss calculation
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = K.models.load_model(model_path)
        self.alpha = alpha

        # adding inputs [A, P. N]
        A = K.Input(shape=(96, 96, 3))
        P = K.Input(shape=(96, 96, 3))
        N = K.Input(shape=(96, 96, 3))
        inputs = [A, P, N]

        X_a = self.base_model(A)
        X_p = self.base_model(P)
        X_n = self.base_model(N)
        encoded_input = [X_a, X_p, X_n]

        decoded = TripletLoss(alpha=alpha)(encoded_input)
        decoder = K.models.Model(inputs, decoded)

        self.training_model = decoder
        self.training_model.compile(optimizer='Adam')
        self.training_model.save

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):

        history = self.training_model.fit(triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=verbose,
                                          validation_split=validation_split)
        return history

    def save(self, save_path):
        """
        Save model
        Args:
            save_path: the path to save the model
        Returns: saved model
        """
        K.models.save_model(self.base_model, save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        F1 score
        Args:
            y_pred: numpy.ndarray of shape (m,)
                    containing the correct labels
        Returns:  the f1 score
        """
        predicted = y_pred
        actual = y_true
        TP = np.count_nonzero(predicted * actual)
        TN = np.count_nonzero((predicted - 1) * (actual - 1))
        FP = np.count_nonzero(predicted * (actual - 1))
        FN = np.count_nonzero((predicted - 1) * actual)

        if TP + FP == 0:
            return 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            return 0
        else:
            recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Accuracy
        Args:
            y_pred: numpy.ndarray of shape (m,)
                    containing the correct labels
        Returns:  the accuracy
        """
        predicted = y_pred
        actual = y_true
        TP = np.count_nonzero(predicted * actual)
        TN = np.count_nonzero((predicted - 1) * (actual - 1))
        FP = np.count_nonzero(predicted * (actual - 1))
        FN = np.count_nonzero((predicted - 1) * actual)

        true_values = TP + TN
        all_values = TP + FP + FN + TN
        accuracy = true_values / all_values
        return accuracy

    @staticmethod
    def distance(emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

    def best_tau(self, images, identities, thresholds):
        """
        Calculate best tau
        Args:
            images: numpy.ndarray of shape (m, n, n, 3) containing
                    the aligned images for testing
            identities: list containing the identities of each
                        image in images
            thresholds: 1D numpy.ndarray of distance thresholds
                        (tau) to test
        Returns: (tau, f1, acc)
        """
        embedded = np.zeros((images.shape[0], 128))

        for i, img in enumerate(images):
            embedded[i] = self.base_model.predict(np.expand_dims(img, axis=0))[0]

        distances = []
        identical = []

        num = len(identities)

        for i in range(num - 1):
            for j in range(i + 1, num):
                distances.append(self.distance(embedded[i], embedded[j]))
                identical.append(1 if identities[i] == identities[j] else 0)

        distances = np.array(distances)
        identical = np.array(identical)

        f1_scores = [self.f1_score(identical, distances < t) for t in thresholds]
        acc_scores = [self.accuracy(identical, distances < t) for t in thresholds]

        f1_idx = np.argmax(f1_scores)
        opt_f1 = f1_scores[f1_idx]
        # Threshold at maximal F1 score
        tau = thresholds[f1_idx]
        # Accuracy at maximal F1 score
        opt_acc = acc_scores[f1_idx]
        return tau, opt_f1, opt_acc
