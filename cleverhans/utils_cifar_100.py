from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pickle
import sys
from tensorflow import one_hot
import warnings

from . import utils


def data_cifar_100(trainfile='data/train', testfile='data/test', label_type=b'fine_labels',
                   train_start=0, train_end=50000,
                   test_start=0, test_end=10000):
    """
    Load and preprocess CIFAR-100 dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    with open(trainfile, 'rb') as f:
        train_dict = pickle.load(f, encoding='bytes')
    with open(testfile, 'rb') as f:
        test_dict = pickle.load(f, encoding='bytes')

    X_train = (np.array(train_dict[b'data'], dtype=float) / 255.0).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
    Y_train = np.eye(100, dtype=float)[train_dict[label_type]]
    X_test = (np.array(test_dict[b'data'], dtype=float) / 255.0).reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    Y_test = np.eye(100, dtype=float)[test_dict[label_type]]

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test