from sklearn.utils import shuffle
from scipy.io import loadmat
import sys
import random
import math
import numpy as np
import datetime

start = datetime.datetime.now()

# read data
mnist = loadmat(sys.argv[1])
test_file = sys.argv[2]


eta = 0.1
X, Y = mnist['data'][:, :60000].T / 255., mnist['label'][:, :60000].T
dev_x, dev_y = mnist['data'][:, -10000:].T / 255., mnist['label'][:, -10000:].T
x = np.array([ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]])
y = np.array([ey for ey in Y if ey in [0, 1, 2, 3]])
dev_x = np.array([ex for ex, ey in zip(dev_x, dev_y) if ey in [0, 1, 2, 3]])
dev_y = np.array([ey for ey in dev_y if ey in [0, 1, 2, 3]])


# suffle examples
x, y = shuffle(x, y, random_state=1)

EPOCHS = 100000
ETA = 0.1
LAMBDA = 0.1
categories = 4
image_size = 784
random_classifier = 5


def binary_svm(train_x, train_y, label):
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
            w = (1 - eta * LAMBDA) * w + eta * target * data
        else:
            w = (1 - eta * LAMBDA) * w

    return w


def accuracy_svm(train_x, train_y, w, label):
    counter = 0.0
    train_y = [1 if x == label else -1 for x in train_y]
    for sample, target in zip(train_x, train_y):
        pred = np.sign(sample.dot(w))
        if pred == target:
            counter += 1
    return counter/len(train_y)

def accuracy(x, y, func, M, params):
    correct = 0.0
    total = 0.0
    for sample, target in zip(x, y):
        pred = func(sample, M, params)
        if pred == target:
            correct += 1
        total += 1
    acc = correct / total
    return acc

def predict(x, func, M, params, fileName):
    output = []
    for sample in x:
        output.append(func(sample, M, params))

    np.savetxt(fileName, output, fmt='%d', delimiter='\n')


def hamming_predict(x, M, params):
    f = x.dot(params.T)
    arr = np.zeros(categories)
    for r in range(categories):
        counter = 0.0
        for s in range(len(f)):
            counter += (1 - np.sign(f[s] * M[r][s])) / 2
        arr[r] = counter
    return np.argmin(arr)


def loss_predict(x, M, params):
    f = x.dot(params.T)
    arr = np.zeros(categories)
    for r in range(categories):
        counter = 0.0
        for s in range(len(f)):
            counter += max(0, 1 - f[s] * M[r][s])
        arr[r] = counter
    return np.argmin(arr)



if __name__ == '__main__':


    #One Vs All
    print('One vs all')
    one_vs_all = np.zeros((categories, image_size))
    for i in range(categories):
        one_vs_all[i] = binary_svm(x, y, i)


    OVA_Matrix = -np.ones((categories, one_vs_all.shape[0]))
    for i in range(4):
        OVA_Matrix[i][i] = 1


    #All Pairs
    print('All pairs')
    classifier_number = int(categories * (categories - 1) / 2)

    idx0_1 = [i for i, label in enumerate(y) if label == 0 or label == 1]
    idx0_2 = [i for i, label in enumerate(y) if label == 0 or label == 2]
    idx0_3 = [i for i, label in enumerate(y) if label == 0 or label == 3]
    idx1_2 = [i for i, label in enumerate(y) if label == 1 or label == 2]
    idx1_3 = [i for i, label in enumerate(y) if label == 1 or label == 3]
    idx2_3 = [i for i, label in enumerate(y) if label == 2 or label == 3]


    all_pairs = np.zeros((classifier_number, image_size))

    dict = {0: '0_1', 1: '0_2', 2: '0_3', 3: '1_2', 4: '1_3', 5:'2_3'}


    all_pairs[0] = binary_svm(x[idx0_1], y[idx0_1], 0)
    all_pairs[1] = binary_svm(x[idx0_2], y[idx0_2], 0)
    all_pairs[2] = binary_svm(x[idx0_3], y[idx0_3], 0)
    all_pairs[3] = binary_svm(x[idx1_2], y[idx1_2], 1)
    all_pairs[4] = binary_svm(x[idx1_3], y[idx1_3], 1)
    all_pairs[5] = binary_svm(x[idx2_3], y[idx2_3], 2)

    
    AP_Matrix = np.zeros((categories, classifier_number))

    for key, value in dict.items():
        right = int(value[0])
        wrong = int(value[-1])
        AP_Matrix[right][key] = 1
        AP_Matrix[wrong][key] = -1



    #Random
    random_params = np.zeros((random_classifier, image_size))

    Random_Matrix = np.zeros((categories, random_classifier))
    test = []
    for i in range(random_classifier):
        right, wrong = np.random.choice(range(4), 2, replace=False)
        while (right, wrong) in test or (wrong, right) in test:
            right, wrong = np.random.choice(range(4), 2, replace=False)
        Random_Matrix[right][i] = 1
        Random_Matrix[wrong][i] = -1
        test.append((right, wrong))
        ids = [j for j, label in enumerate(y) if label == right or label == wrong]
        random_params[i] = binary_svm(x[ids], y[ids], right)


    train_svm = datetime.datetime.now() - start
    print("Training SVM : " + str(train_svm) + "\n")


    #Accuracy

    #One-vs-all
    acc_hamming_ovs = accuracy(dev_x, dev_y, hamming_predict, OVA_Matrix, one_vs_all)
    print('Accuracy hamming ovs: ' + str(acc_hamming_ovs))

    acc_loss_ovs = accuracy(dev_x, dev_y, loss_predict, OVA_Matrix, one_vs_all)
    print('Accuracy loss ovs: ' + str(acc_loss_ovs))

    #All-pairs
    acc_hamming_ap = accuracy(dev_x, dev_y, hamming_predict, AP_Matrix, all_pairs)
    print('Accuracy hamming ap: ' + str(acc_hamming_ap))

    acc_loss_ap = accuracy(dev_x, dev_y, loss_predict, AP_Matrix, all_pairs)
    print('Accuracy loss ap: ' + str(acc_loss_ap))


    #Random matrix
    acc_hamming_rnd = accuracy(dev_x, dev_y, hamming_predict, Random_Matrix, random_params)
    print('Accuracy hamming random: ' + str(acc_hamming_rnd))

    acc_loss_rnd = accuracy(dev_x, dev_y, loss_predict, Random_Matrix, random_params)
    print('Accuracy loss random: ' + str(acc_loss_rnd))


    evaluate = datetime.datetime.now() - train_svm
    print("Evaluate accuracy : " + str(evaluate) + "\n")


    test_data = np.loadtxt(test_file)
    predict(test_data, hamming_predict, OVA_Matrix, one_vs_all, 'test.onevall.ham.pred')
    predict(test_data, hamming_predict, AP_Matrix, all_pairs, 'test.allpairs.ham.pred')
    predict(test_data, hamming_predict, Random_Matrix, random_params, 'test.randm.ham.pred')
    predict(test_data, loss_predict, OVA_Matrix, one_vs_all, 'test.onevall.loss.pred')
    predict(test_data, loss_predict, AP_Matrix, all_pairs, 'test.allpairs.loss.pred')
    predict(test_data, loss_predict, Random_Matrix, random_params, 'test.randm.loss.pred')

    predict = datetime.datetime.now() - evaluate

    print("Predict : " + str(predict) + "\n")