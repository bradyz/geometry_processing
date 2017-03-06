import pickle
import time

import numpy as np

from matplotlib import pyplot as plt

from sklearn import linear_model

from geometry_processing.train_cnn.classify_keras import load_model_vgg
from geometry_processing.globals import (TRAIN_DIR, VALID_DIR, NUM_CLASSES,
        IMAGE_MEAN, IMAGE_STD, SAVE_FILE, FC2_MEAN, FC2_STD)
from geometry_processing.utils.helpers import (get_data, samplewise_normalize,
        extract_layer)
from geometry_processing.utils.custom_datagen import GroupedDatagen


NUM_BATCHES = 150


def entropy(x):
    return -np.sum(x * np.log(x))


def fc2_normal_entropy(x, y, fc2_layer, fc2_normalize, softmax_layer):
    x_fc2 = fc2_layer.predict(x)
    x_fc2_normal = np.apply_along_axis(fc2_normalize, 1, x_fc2)

    x_softmax = softmax_layer.predict(x)
    x_entropy = [(entropy(x_softmax[i]), i) for i in range(x_softmax.shape[0])]

    y = np.argmax(y, axis=1)

    return x_fc2_normal, x_entropy, y


def evaluate_using_k(fc2_layer, softmax_layer, train_group, valid_group,
        top_k=3, batch_size=32, log_file=None):
    # Normalize the activation feature vectors.
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)

    # SGDClassifier with hinge loss and l2 is an SVM.
    svm = linear_model.SGDClassifier(penalty='l2', alpha=0.0002, loss='hinge')

    print("Top k: %d" % top_k)

    if log_file is not None:
        log_file.write("batch,train_acc,val_acc\n")
        log_file.flush()

    for t, batch in enumerate(zip(train_group.generate(batch_size=batch_size),
                                  valid_group.generate(batch_size=batch_size))):
        if t >= NUM_BATCHES:
            break
        print("Batch %d/%d" % (t+1, NUM_BATCHES))

        # Images are mean centered and normalized.
        train, valid = batch

        # TODO: don't hardcode these.
        examples = np.zeros((batch_size, 2048))
        labels = np.zeros((batch_size))

        examples_valid = np.zeros((batch_size, 2048))
        labels_valid = np.zeros((batch_size))

        for i in range(batch_size):
            # Prep data for training.
            x_fc, x_sm, y = fc2_normal_entropy(train[0][i],
                    train[1][i], fc2_layer, fc2_normalize, softmax_layer)

            x_fc_valid, x_sm_valid, y_valid = fc2_normal_entropy(valid[0][i],
                    valid[1][i], fc2_layer, fc2_normalize, softmax_layer)

            # Sort by entropy.
            x_sm.sort(key=lambda entropy_index: entropy_index[0])
            x_sm_valid.sort(key=lambda entropy_index: entropy_index[0])

            # Create a chunk using only the most confident views.
            top_k_fc = np.zeros((top_k, x_fc.shape[1]))
            top_k_fc_valid = np.zeros((top_k, x_fc_valid.shape[1]))

            # Grab the top k.
            for j in range(top_k):
                top_k_fc[j] = x_fc[x_sm[j][1]]
                top_k_fc_valid[j] = x_fc_valid[x_sm_valid[j][1]]

            # Form the activation vector, which is element wise max of top k.
            examples[i] = np.max(top_k_fc, axis=0)
            labels[i] = y[0]

            examples_valid[i] = np.max(top_k_fc_valid, axis=0)
            labels_valid[i] = y_valid[0]

        # Train on batch.
        svm.partial_fit(examples, labels, classes=range(NUM_CLASSES))

        if log_file is not None:
            log_file.write("%d,%.4f,%.4f\n" %
                    (t, svm.score(examples, labels),
                        svm.score(examples_valid, labels_valid)))
            log_file.flush()

        if t > 0 and t % 25 == 0:
            # Save trained svm.
            svm_pickle = "svm_top_k_%d.pkl" % top_k
            with open(svm_pickle, 'wb') as fid:
                pickle.dump(svm, fid)
            print("Saved to %s." % svm_pickle)


if __name__ == '__main__':
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    train_group = GroupedDatagen(TRAIN_DIR, preprocess=img_normalize)
    valid_group = GroupedDatagen(VALID_DIR, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model_vgg(SAVE_FILE)
    fc2_layer = extract_layer(model, 'fc2')
    softmax_layer = extract_layer(model, 'predictions')

    # Sort by top K minimized entropy.
    for k in [1, 3, 5, 7]:
        with open("top_%d.log" % k, "w") as log_file:
            evaluate_using_k(fc2_layer, softmax_layer,
                    train_group, valid_group, top_k=k, log_file=log_file)
