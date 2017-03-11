import os
import pickle

import numpy as np

from sklearn.metrics import confusion_matrix

from geometry_processing.globals import (VALID_DIR, NUM_CLASSES, SAVE_FILE,
        IMAGE_MEAN, IMAGE_STD, FC2_MEAN, FC2_STD, PACKAGE_PATH, LOG_FILE)
from geometry_processing.utils.helpers import (plot_confusion_matrix, get_data,
        samplewise_normalize, extract_layer)
from geometry_processing.train_cnn.classify_keras import load_model
from geometry_processing.utils.custom_datagen import GroupedDatagen
from geometry_processing.classification.linear_classifier import (get_top_k,
        min_entropy, fc2_normal)


LOG_FD = open(os.path.join(PACKAGE_PATH, "logs", "upper.log"), "w")
NUM_SAMPLES = 1000


def get_class_labels(data_generator):
    # Something like ['bathtub', 'desk', ...].
    class_labels = data_generator.class_indices
    class_labels = list(sorted(class_labels, key=lambda x: class_labels[x]))
    return class_labels


def add_to_confusion_matrix(model, matrix, data_generator, logits=False):
    # Number of samples tested.
    total = 0

    # Flow for NUM_SAMPLES samples.
    for images, labels in data_generator:
        if total > NUM_SAMPLES:
            break

        # Get predictions for this batch.
        y_pred = model.predict(images)
        if logits:
            y_pred = np.argmax(y_pred, axis=1)

        y_true = np.argmax(labels, axis=1)

        # Aggregate confusion matrix.
        matrix += confusion_matrix(y_true, y_pred, labels=range(10))

        # Logging stuff.
        total += images.shape[0]
        print("%d / %d samples processed." % (total, NUM_SAMPLES))

    np.save(SAVE_FILE, matrix)


def evaluate_cnn():
    # Set up crap.
    data_generator = get_data(VALID_DIR)
    model = load_model(SAVE_FILE)

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    plot_confusion_matrix(matrix, get_class_labels(data_generator))


def evaluate_svm(svm, fc2_layer, softmax_layer, datagen, batch_size=64, top_k=3,
        out_file_path=None):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    # Total samples evaluated.
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)

    # Write to log.
    LOG_FD.write("k: %d\n" % top_k)
    LOG_FD.flush()

    total = 0
    # Images are mean centered and normalized.
    for batch in datagen.generate(batch_size=batch_size):
        if total >= NUM_SAMPLES:
            break
        LOG_FD.write("Samples Processed %d/%d\n" % (total, NUM_SAMPLES))
        LOG_FD.flush()

        # TODO: don't hardcode these.
        x = np.zeros((batch_size, 2048))
        y_true = np.zeros((batch_size))

        for i in range(batch_size):
            # Train.
            x_i = batch[0][i]
            y_i = batch[1][i]

            x_fc = fc2_normal(x_i, fc2_layer, fc2_normalize)
            x_entropy = min_entropy(x_i, softmax_layer)
            x_top_fc = get_top_k(x_fc, x_entropy, top_k)

            # Form the activation vector, which is element wise max of top k.
            x[i] = np.max(x_top_fc, axis=0)

            # All labels along axis 0 should be the same.
            y_true[i] = np.argmax(y_i, axis=1)[0]

        # Train on batch.
        y_pred = svm.predict(x)
        matrix += confusion_matrix(y_true, y_pred, labels=range(10))

        total += batch[0].shape[0]

    if out_file_path:
        LOG_FD.write("Saving to %s.\n" % out_file_path)
        LOG_FD.flush()
        np.save(out_file_path, matrix)

        accuracy = np.sum(np.diag(matrix)) / np.sum(matrix, axis=(0,1))
        LOG_FD.write("Accuracy %.4f\n" % accuracy)
        LOG_FD.flush()


def gather_svm():
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    test_group = GroupedDatagen(VALID_DIR, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model(SAVE_FILE)
    fc2_layer = extract_layer(model, 'fc2')
    softmax_layer = extract_layer(model, 'predictions')

    for i in [1, 3, 5, 7]:
        root = PACKAGE_PATH
        svm_path = os.path.join(root, "cache", "svm_top_k_%d.pkl" % i)
        out_path = os.path.join(root, "cache", "svm_top_k_%d_confusion.npy" % i)

        with open(svm_path, "rb") as fd:
            svm = pickle.load(fd)

        evaluate_svm(svm, fc2_layer, softmax_layer, test_group,
                top_k=i, out_file_path=out_path)


if __name__ == '__main__':
    data_generator = get_data(VALID_DIR)

    for k in [1, 3, 5, 7, 10, 15, 20, 25]:
        print("K: %d" % k)
        matrix = np.load(os.path.join(PACKAGE_PATH, "cache", "svm_top_k_%d_confusion.npy" % k))
        plot_confusion_matrix(matrix, get_class_labels(data_generator),
                normalize=True, title="Confusion @ K=%d" % k)
