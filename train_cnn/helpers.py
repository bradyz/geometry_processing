"""
This file assumes keras ordering is set to 'tf'.
"""

import cv2

from matplotlib import pyplot as plt

import numpy as np

from keras.callbacks import Callback


class ManualInspection(Callback):
    def __init__(self, model):
        self.model = model

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass


def show(image):
    plt.imshow(image, interpolation='none')
    plt.show()
    return


def view_filters(weights, number_to_show=10, axis=0):
    filter_size_x, filter_size_y, dimensions, number = weights.shape
    print("Filter size: %d by %d by %d" % (filter_size_x, filter_size_y,
					   dimensions))

    assert axis < dimensions, "Invalid axis."

    for i in range(number_to_show):
        show(weights[:, :, axis, i])


def to_rgb(greyscale_image):
    return cv2.cvtColor(greyscale_image, cv2.COLOR_GRAY2RGB)


def convert_greyscale_to_rgb(data):
    samples, image_x, image_y, channels = data.shape

    assert channels == 1, "Data must be greyscale."

    result = np.empty([samples, image_x, image_y, 3])
    for i in range(samples):
        result[i] = to_rgb(data[i])
    return result


def resize_dataset(data, output_x, output_y):
    samples, image_x, image_y, channels = data.shape

    result = np.empty([samples, output_x, output_y, channels])
    for i in range(samples):
        result[i] = cv2.resize(data[i], (output_x, output_y))
    return result
