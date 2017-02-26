"""
This file assumes keras ordering is set to 'tf'.
"""

import itertools

import cv2

from matplotlib import pyplot as plt

import numpy as np

from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

from geometry_processing.globals import IMAGE_SIZE, BATCH


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


def get_data(data_path, batch=BATCH):
    data_datagen = ImageDataGenerator(samplewise_center=True,
            samplewise_std_normalization=True)
    data_generator = data_datagen.flow_from_directory(
            data_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=batch)
    return data_generator


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def test_from_path(model, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = np.reshape(image, (1, 224, 224, 3))
    return test_from_image(model, image)


# Image must be of size (1, 224, 224, 3).
def test_from_image(model, image):
    prediction = model.predict(image)
    class_name = np.argmax(prediction, axis=1)
    return prediction, class_name
