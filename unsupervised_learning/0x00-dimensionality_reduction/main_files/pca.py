#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
pca = __import__('1-pca').pca

X = np.loadtxt("mnist2500_X.txt")
labels = np.loadtxt("mnist2500_labels.txt")
Y = pca(X, 2)
plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.colorbar()
plt.title('PCA')
plt.show()