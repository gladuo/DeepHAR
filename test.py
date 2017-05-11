# coding=utf-8

import os
import time

import numpy as np
from keras import models

from utils.load_dataset import load_dataset
from utils.sliding_window import sliding_window, opp_sliding_window
from config import *


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def prepare_data():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NUM_SENSOR_CHANNELS))
    return X_test, y_test


def test():
    # Prepare data
    X_test, y_test = prepare_data()

    # Load model
    json_string = open('./runs/{}/model_pickle.json'.format(str(TEST_MODEL_NUMBER)), 'r').read()
    model = models.model_from_json(json_string)

    # Load weights
    weights_file_list = sorted(os.listdir('./runs/{}'.format(str(TEST_MODEL_NUMBER))))
    weights_file = weights_file_list[0] if weights_file_list[0] is 'model_1_weights_sub.h5' else weights_file_list[-1]
    model.load_weights('./runs/{}/{}'.format(str(TEST_MODEL_NUMBER), weights_file))

    # Test model
    print("Processing {0} instances in mini-batches of {1}".format(X_test.shape[0], BATCH_SIZE))
    test_pred = np.empty(0)
    test_true = np.empty(0)
    start_time = time.time()
    for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE):
        inputs, targets = batch
        y_pred = model.predict(inputs, batch_size=BATCH_SIZE)
        test_pred = np.append(test_pred, map(lambda i: i.argmax(), y_pred), axis=0)
        test_true = np.append(test_true, targets, axis=0)

    print "||Results||"
    print "\tTook {:.3f}s.".format(time.time() - start_time)
    import sklearn.metrics as metrics
    print "\tTest fscore:\t{:.4f} ".format(metrics.f1_score(test_true, test_pred, average='weighted'))


if __name__ == '__main__':
    test()
