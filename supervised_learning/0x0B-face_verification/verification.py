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
        embedded = np.zeros((images.shape[0], 128))

        for i, img in enumerate(images):
            embedded[i] = self.base_model.predict(np.expand_dims(img, axis=0))[0]
        return embedded

    def verify(self, image, tau=0.5):
