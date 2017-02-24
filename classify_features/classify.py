import sys
import os
import itertools

from sklearn import svm
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

MODEL_PATH = '/Users/bradyzhou/code/geometry_processing/train_cnn'
sys.path.append(MODEL_PATH)

import classify_keras

TRAIN_DIR = '/Users/bradyzhou/code/data/ModelNetViewpoints/train/'
VALID_DIR = '/Users/bradyzhou/code/data/ModelNetViewpoints/test/'
IMAGE_SIZE = 224
BATCH = 64


def get_data(data_path):
    data_datagen = ImageDataGenerator(samplewise_center=True,
            samplewise_std_normalization=True)
    data_generator = data_datagen.flow_from_directory(
            data_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH)
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


if __name__ == '__main__':
    data_generator = get_data(VALID_DIR)

    class_labels = data_generator.class_indices
    class_labels = list(sorted(class_labels, key=lambda x: class_labels[x]))

    model = classify_keras.load_model_vgg()
    matrix = None

    for images, labels in data_generator:
        predictions = model.predict(images)

        labels = np.argmax(labels, axis=1)
        predictions = np.argmax(predictions, axis=1)

        if matrix is None:
            matrix = confusion_matrix(labels, predictions, labels=range(10))
        else:
            matrix += confusion_matrix(labels, predictions, labels=range(10))

    np.save('confusion_matrix.npy', matrix)
    plot_confusion_matrix(matrix, class_labels)
