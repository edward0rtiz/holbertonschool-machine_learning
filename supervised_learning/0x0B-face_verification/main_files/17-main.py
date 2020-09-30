#!/usr/bin/env python3

import numpy as np
from verification import FaceVerification
from utils import load_images

images, _ = load_images('HBTNaligned', as_array=True)

np.random.seed(0)
database = np.random.randn(5, 128)
identities = ['Holberton', 'school', 'is', 'the', 'best!']
fv = FaceVerification('models/trained_fv.h5', database, identities)
embs = fv.embedding(images)
print(embs.shape)