# coding=utf-8

import os
import time
import numpy as np

from keras.layers import Input, Convolution2D, GRU, LSTM, Dense, Permute, Reshape, Flatten, ELU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from utils.load_dataset import load_dataset
from utils.sliding_window import sliding_window, opp_sliding_window
from config import *


def train():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')

    assert NUM_SENSOR_CHANNELS == X_train.shape[1]

    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NUM_SENSOR_CHANNELS))
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NUM_SENSOR_CHANNELS))

    # network
    inputs = Input(shape=(1, SLIDING_WINDOW_LENGTH, NUM_SENSOR_CHANNELS))
    conv1 = ELU()(Convolution2D(NUM_FILTERS, FILTER_SIZE, 1, border_mode='valid', init='normal', activation='relu')(inputs))
    conv2 = ELU()(Convolution2D(NUM_FILTERS, FILTER_SIZE, 1, border_mode='valid', init='normal', activation='relu')(conv1))
    conv3 = ELU()(Convolution2D(NUM_FILTERS, FILTER_SIZE, 1, border_mode='valid', init='normal', activation='relu')(conv2))
    conv4 = ELU()(Convolution2D(NUM_FILTERS, FILTER_SIZE, 1, border_mode='valid', init='normal', activation='relu')(conv3))
    # permute1 = Permute((2, 1, 3))(conv4)
    reshape1 = Reshape((8, NUM_FILTERS * NUM_SENSOR_CHANNELS))(conv4)
    gru1 = GRU(NUM_UNITS_LSTM, return_sequences=True, consume_less='mem')(reshape1)
    gru2 = GRU(NUM_UNITS_LSTM, return_sequences=False, consume_less='mem')(gru1)
    outputs = Dense(NUM_CLASSES, activation='softmax')(gru2)

    model = Model(input=inputs, output=outputs)
    # Save checkpoints
    timestamp = str(int(time.time()))
    os.mkdir('./runs/%s/' % timestamp)
    checkpoint = ModelCheckpoint('./runs/%s/weights.{epoch:03d}-{val_acc:.4f}.hdf5' % timestamp, monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max')

    json_string = model.to_json()
    open('./runs/%s/model_pickle.json' % timestamp, 'w').write(json_string)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHES, verbose=1, callbacks=[checkpoint],
              validation_data=(X_test, y_test))  # starts training

    model.save_weights('./runs/%s/model_1_weights_sub.h5' % timestamp)

    # score = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    # print score

if __name__ == '__main__':
    train()
