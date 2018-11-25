from sklearn.utils import shuffle
from scipy.io import loadmat
import sys
import random
import math
import numpy as np

# read data
mnist = loadmat(sys.argv[1])
eta = 0.1
X, Y = mnist['data'][:, :60000].T / 255., mnist['label'][:, :60000].T
x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
y = [ey for ey in Y if ey in [0, 1, 2, 3]]

# suffle examples
x, y = shuffle(x, y, random_state=1)

EPOCHS = 10
ETA = 0.1

def binary_svm(train_x, train_y, eta, gamma):
    indexes = list(range(len(train_y)))
    image_size = len(train_x[0])
    w = np.random.rand(image_size)
    for i in range(1, EPOCHS):
        index = random.choice(indexes)
        data, label = np.array(x[index]), np.array(y[index])
        eta /= math.sqrt(i)
        if 1 - label * data.dot(w) > 0:
            w = (1 - eta * gamma) * w + eta * label * data
        else:
            w = (1 - eta * gamma) * w

    return w

