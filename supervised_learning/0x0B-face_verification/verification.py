#!/usr/bi/env python3
"""
Verification face recognition
"""

import tensorflow.keras as K
import tensorflow as tf
import numpy as np


class FaceVerification:
    """
    Class verification
    """
    def __init__(self, model_path, database, identities):
        with K.utils.CustomObjectScope({'tf': tf}):
            self.model = K.models.load_model(model_path)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
        embedding
        Args:
            images: Images to be embedded
        Returns: embedded images
        """
        embedded = np.zeros((images.shape[0], 128))

        for i, img in enumerate(images):
            embedded[i] = self.model.predict(np.expand_dims(img, axis=0))[0]
        return embedded

    def verify(self, image, tau=0.5):
        """
        Verify tay
        Args:
            image: input images
            tau: tau for best distances between vectors
        Returns: TAU
        """

        def distance(emb1, emb2):
            """
            distance
            Args:
                emb1: emb1
                emb2: emb2
            Returns: distance

            """
            return np.sum(np.square(emb1 - emb2))

        distances = []
        encoding = self.model.predict(np.expand_dims(image, axis=0))[0]

        num = len(self.identities)
        for i in range(num):
            distances.append(distance(encoding, self.database[i]))

        distances = np.array(distances)
        min_dist = np.argmin(distances)

        if distances[min_dist] < tau:
            return self.identities[min_dist], distances[min_dist]
        else:
            return None, None
