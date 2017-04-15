# coding=utf-8

import numpy as np
import cPickle as cp

from sliding_window import sliding_window


def load_dataset(filename='data/oppChallenge_gestures.data'):
    """
    
    :param filename: 
    :return: Preprocessed Opportunity dataset
    """
    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    load_dataset()
