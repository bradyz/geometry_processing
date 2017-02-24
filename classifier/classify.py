import numpy as np

from sklearn.metrics import confusion_matrix

from geometry_processing.utils.helpers import (TRAIN_DIR, VALID_DIR,
        BATCH, NUM_CLASSES, get_data, plot_confusion_matrix)
from geometry_processing.train_cnn.classify_keras import load_model_vgg


NUM_SAMPLES = 1000


if __name__ == '__main__':
    data_generator = get_data(VALID_DIR)

    # Something like ['bathtub', 'desk', ...].
    class_labels = data_generator.class_indices
    class_labels = list(sorted(class_labels, key=lambda x: class_labels[x]))

    # Set up model.
    model = load_model_vgg()

    # Initialize confusion matrix.
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

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
    plot_confusion_matrix(matrix, class_labels)
