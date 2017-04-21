# coding=utf-8

from keras import models
from keras.utils.vis_utils import plot_model

from config import *


def show_plot():
    # Load model
    json_string = open('./runs/{}/model_pickle.json'.format(str(TEST_MODEL_NUMBER)), 'r').read()
    model = models.model_from_json(json_string)

    # Show model
    plot_model(model, to_file='%s.png' % TEST_MODEL_NUMBER)


if __name__ == '__main__':
    show_plot()
