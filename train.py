# coding=utf-8

import os
import time
import numpy as np

import keras
from keras.layers import Input, Conv2D, GRU, LSTM, Dense, Dropout, Permute, Reshape, Flatten, ELU
from keras.layers.merge import concatenate, add, dot
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1, l2
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

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
    conv1 = ELU()(
        Conv2D(filters=NUM_FILTERS, kernel_size=(1, FILTER_SIZE), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(inputs))
    conv2 = ELU()(
        Conv2D(filters=NUM_FILTERS, kernel_size=(1, FILTER_SIZE), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(conv1))
    conv3 = ELU()(
        Conv2D(filters=NUM_FILTERS, kernel_size=(1, FILTER_SIZE), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(conv2))
    conv4 = ELU()(
        Conv2D(filters=NUM_FILTERS, kernel_size=(1, FILTER_SIZE), strides=(1, 1), padding='same',
               kernel_initializer='random_normal', data_format='channels_last')(conv3))
    reshape1 = Reshape((SLIDING_WINDOW_LENGTH, NUM_FILTERS * 1))(conv4)
    dropout1 = Dropout(DROPOUT_RATE)(reshape1)
    gru1 = GRU(NUM_UNITS_LSTM, return_sequences=True, implementation=2)(dropout1)
    dropout2 = Dropout(DROPOUT_RATE)(gru1)
    gru2 = GRU(NUM_UNITS_LSTM, return_sequences=False, implementation=2)(dropout2)  # implementation=2 for GPU
    dropout3 = Dropout(DROPOUT_RATE)(gru2)

    outputs = Dense(NUM_CLASSES, activation=K.softmax, activity_regularizer=l2())(dropout3)

    model = Model(inputs=inputs, outputs=outputs)
    # Save checkpoints
    timestamp = str(int(time.time()))
    os.mkdir('./runs/%s/' % timestamp)
    checkpoint = ModelCheckpoint('./runs/%s/weights.{epoch:03d}-{val_acc:.4f}.hdf5' % timestamp, monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max')
    # Save model networks
    json_string = model.to_json(indent=4)
    open('./runs/%s/model_pickle.json' % timestamp, 'w').write(json_string)

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # rmsprop = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-8)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHES, verbose=1, callbacks=[checkpoint],
              validation_data=(X_test, y_test))  # starts training
    model.save_weights('./runs/%s/model_1_weights_sub.h5' % timestamp)


if __name__ == '__main__':
    train()
