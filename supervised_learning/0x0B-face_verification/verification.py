#!/usr/bi/env python3
"""
Verification face recognition
"""

import tensorflow.keras as K
import tensorflow as tf

class FaceVerification:
    def __init__(self, model, database, identities):
        with K.utils.CustomObjectScope({'tf': tf}):
            self.model = K.models.load_model(model)
        self.database = database
        self.identities = identities

    def embedding(self, images):
        embeddings = self.model.predict(images)
        return embeddings

    def verify(self, image, tau=0.5):
        return identity, distance