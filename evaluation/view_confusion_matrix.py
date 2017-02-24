import numpy as np

from sklearn.metrics import confusion_matrix

from geometry_processing.utils.helpers import (TRAIN_DIR, VALID_DIR,
        BATCH, NUM_CLASSES, get_data, plot_confusion_matrix)
from geometry_processing.train_cnn.classify_keras import load_model_vgg


NUM_SAMPLES = 1000
USE_SAVE = True
SAVE_FILE = 'confusion_matrix.npy'


def get_class_labels(data_generator):
    # Something like ['bathtub', 'desk', ...].
    class_labels = data_generator.class_indices
    class_labels = list(sorted(class_labels, key=lambda x: class_labels[x]))
    return class_labels


def add_to_confusion_matrix(matrix, data_generator):
    # Set up model.
    model = load_model_vgg()

    # Number of samples tested.
    total = 0

    # Flow for NUM_SAMPLES samples.
    for images, labels in data_generator:
        if total > NUM_SAMPLES:
            break

        # Get predictions for this batch.
        softmax = model.predict(images)

        # Throw away other predictions.
        y_true = np.argmax(labels, axis=1)
        y_pred = np.argmax(softmax, axis=1)

        # Aggregate confusion matrix.
        matrix += confusion_matrix(y_true, y_pred, labels=range(10))

        total += BATCH
        print("%-4.2f%%: %d samples processed." % (100*total/NUM_SAMPLES, total))

    np.save('confusion_matrix.npy', matrix)


if __name__ == '__main__':
    # Get data.
    data_generator = get_data(VALID_DIR)

    # Initialize confusion matrix.
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    if USE_SAVE:
        matrix = np.load(SAVE_FILE)

    class_labels = get_class_labels(data_generator)
    plot_confusion_matrix(matrix, class_labels)
