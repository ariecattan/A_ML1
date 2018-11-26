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
x = np.array([ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]])
y = np.array([ey for ey in Y if ey in [0, 1, 2, 3]])

# suffle examples
x, y = shuffle(x, y, random_state=1)

EPOCHS = 1000000
ETA = 0.1
LAMBDA = 0.3

def binary_svm(train_x, train_y, lamb, label):
    losses = []
    train_y = [1 if x == label else -1 for x in train_y]
    indexes = list(range(len(train_y)))
    image_size = len(train_x[0])
    w = np.zeros(image_size)
    for i in range(1, EPOCHS):
        index = random.choice(indexes)
        data, target = train_x[index], train_y[index]
        eta = ETA / math.sqrt(i)
        loss =  1 - target * data.dot(w)
        losses.append(loss)
        if loss > 0:
            w = (1 - eta * lamb) * w + eta * target * data
        else:
            w = (1 - eta * lamb) * w

    return w, losses


def one_vs_all(x, y, label):
    train_x = x
    train_y = np.zeros(len(y)) - 1
    id_label = [i for i, lab in enumerate(y) if label==lab]
    train_y[id_label] = 1
    w = binary_svm(train_x, train_y, ETA, LAMBDA)
    return w


def accuracy(train_x, train_y, w, label):
    counter = 0.0
    train_y = [1 if x == label else -1 for x in train_y]
    for sample, target in zip(train_x, train_y):
        pred = np.sign(sample.dot(w))
        if pred == target:
            counter += 1
    return counter/len(train_y)


if __name__ == '__main__':
    idx0_1 = [i for i, label in enumerate(y) if label == 0 or label == 1]
    idx0_2 = [i for i, label in enumerate(y) if label == 1 or label == 2]
    idx0_3 = [i for i, label in enumerate(y) if label == 2 or label == 3]
    idx1_2 = [i for i, label in enumerate(y) if label == 1 or label == 2]
    idx1_3 = [i for i, label in enumerate(y) if label == 1 or label == 3]
    idx2_3 = [i for i, label in enumerate(y) if label == 2 or label == 3]

    idx0 = [i for i, label in enumerate(y) if label == 0]
    idx1 = [i for i, label in enumerate(y) if label == 1]
    idx2 = [i for i, label in enumerate(y) if label == 2]
    idx3 = [i for i, label in enumerate(y) if label == 3]


    params = {}

    train_x, train_y = x[idx0_1], y[idx0_1]
    w, losses = binary_svm(train_x, train_y, LAMBDA, 0)
    print(accuracy(train_x, train_y, w, 0))
