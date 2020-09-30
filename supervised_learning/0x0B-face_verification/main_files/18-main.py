#!/usr/bin/env python3

from matplotlib.pyplot import imread, imsave
import numpy as np
import tensorflow as tf
from utils import load_images
from verification import FaceVerification

database_files = ['HBTNaligned/HeimerRojas.jpg', 'HBTNaligned/MariaCoyUlloa.jpg', 'HBTNaligned/MiaMorton.jpg', 'HBTNaligned/RodrigoCruz.jpg', 'HBTNaligned/XimenaCarolinaAndradeVargas.jpg']
database_imgs = np.zeros((5, 96, 96, 3))
for i, f in enumerate(database_files):
    database_imgs[i] = imread(f)

database_imgs = database_imgs.astype('float32') / 255

with tf.keras.utils.CustomObjectScope({'tf': tf}):
    base_model = tf.keras.models.load_model('models/face_verification.h5')
    database_embs = base_model.predict(database_imgs)

test_img_positive = imread('HBTNaligned/HeimerRojas0.jpg').astype('float32') / 255
test_img_negative = imread('HBTNaligned/KirenSrinivasan.jpg').astype('float32') / 255

identities = ['HeimerRojas', 'MariaCoyUlloa', 'MiaMorton', 'RodrigoCruz', 'XimenaCarolinaAndradeVargas']
fv = FaceVerification('models/face_verification.h5', database_embs, identities)
print(fv.verify(test_img_positive, tau=0.44))
print(fv.verify(test_img_negative, tau=0.44))