"""
This file assumes keras ordering is set to 'tf'.
"""

import itertools

import cv2

from matplotlib import pyplot as plt

import numpy as np

from keras.models import Model
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

from geometry_processing.globals import IMAGE_SIZE


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


def get_data(data_path, batch=64, preprocess=None, shuffle=True):
    data_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    return data_datagen.flow_from_directory(data_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=batch,
            shuffle=shuffle)


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
        plt.text(j, i, format(cm[i, j], '.2f'),
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


def flow_from_directory_statistics(dirname, batch_size=64, num_samples=1000):
    """
    Iteratively calculate mean and std from a dataset too large to fit
    in memory. Uses Welford's Method -
    https://www.johndcook.com/blog/standard_deviation/
    NOTE: there is a bug in the std calculation.

    Args:
        dirname (string) - directory with class labels and data.
        batch_size (int) - number of samples to process in a batch.
        num_samples (int) - terminate early after seeing x batches.
    Returns:
        (tuple), corresponding mean RGB values.
    """
    datagen = get_data(dirname, batch_size)

    mean = np.zeros((3,), dtype=K.floatx())
    running = np.zeros((3,), dtype=K.floatx())
    seen = 0

    for x, _ in datagen:
        # Terminate if datagen is exhausted or sampled sufficiently.
        if datagen.batch_index == 0 and seen != 0:
            break
        elif seen >= num_samples:
            break

        for i in range(x.shape[0]):
            if seen >= num_samples:
                break
            seen += 1
            sample = np.sum(x[i], axis=(0, 1)) / (IMAGE_SIZE ** 2)

            delta = sample - mean
            mean = mean + delta / seen
            running = running + delta * (sample - mean)

    return mean, running / (seen - 1)


def get_precomputed_statistics(directory, num_samples=50):
    # Keras generators use float32 which gives weird values.
    K.set_floatx('float64')

    # Get a clean datagen.
    vanilla_datagen = get_data(directory)

    # Collect a bunch of samples.
    x = np.zeros((num_samples, IMAGE_SIZE, IMAGE_SIZE, 3))

    index = 0
    for x_, _ in vanilla_datagen:
        if index >= num_samples:
            break

        offset = min(num_samples - index, x_.shape[0])
        x[index:index+offset] = x_[:offset]
        index += offset

    # Actually fit the data and compute statistics.
    statistics_datagen = ImageDataGenerator(
            featurewise_std_normalization=True,
            featurewise_center=True)
    statistics_datagen.fit(x)

    print("Dataset path: %s" % directory)
    print("Sample mean: %s" % statistics_datagen.mean)
    print("Sample standard deviation: %s" % statistics_datagen.std)

    return statistics_datagen.mean, statistics_datagen.std


def samplewise_normalize(mean, std):
    return lambda x: (x - mean) / (std + 1e-7)


def extract_layer(full_model, layer):
    intermediate_layer_model = Model(input=full_model.input,
                                     output=full_model.get_layer(layer).output)
    return intermediate_layer_model
