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


NUM_BATCHES = 20
BATCH_SIZE = 64


def entropy(x):
    return -np.sum(x * np.log(x))


def fc2_normal_entropy(x, y, fc2_layer, fc2_normalize, softmax_layer):
    x_fc2 = fc2_layer.predict(x)
    x_fc2_normal = np.apply_along_axis(fc2_normalize, 1, x_fc2)

    x_softmax = softmax_layer.predict(x)
    x_entropy = [(entropy(x_softmax[i]), i) for i in range(x_softmax.shape[0])]

    y = np.argmax(y, axis=1)

    return x_fc2_normal, x_entropy, y


def top_one_svm(fc2, train_datagen, valid_datagen, plot=True):
    # Normalize the activation feature vectors.
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)

    # SGDClassifier with hinge loss and l2 is an SVM.
    svm = linear_model.SGDClassifier(penalty='l2', alpha=0.0001, loss='hinge')

    # Used for logging training details.
    valid_score = list()
    valid_ticks = list()

    train_score = list()
    train_ticks = list()

    if plot:
        plt.ion()
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')

        train_plot = plt.plot(train_ticks, train_score, label='train score')[0]
        valid_plot = plt.plot(valid_ticks, valid_score, label='valid score')[0]
        plt.legend(loc='best')

    for i, batch in enumerate(zip(train_datagen, valid_datagen)):
        if i >= NUM_BATCHES:
            break
        print('Batch #%d - %.2f%% completed.' % (i, i * 100.0 / NUM_BATCHES))

        train_batch, valid_batch = batch

        # Prep data for training.
        x, y = train_batch
        x = fc2.predict(x)
        x = np.apply_along_axis(fc2_normalize, 1, x)
        y = np.argmax(y, axis=1)

        x_valid, y_valid = valid_batch
        x_valid = fc2.predict(x_valid)
        x_valid = np.apply_along_axis(fc2_normalize, 1, x_valid)
        y_valid = np.argmax(y_valid, axis=1)

        # Train on batch.
        svm.partial_fit(x, y, classes=range(NUM_CLASSES))

        # Add accuracy to array.
        train_ticks.append(i)
        train_score.append(svm.score(x, y))
        valid_ticks.append(i)
        valid_score.append(svm.score(x_valid, y_valid))

        print("Training accuracy %.3f" % train_score[-1])
        print("Validation accuracy %.3f" % valid_score[-1])

        # Update plot live.
        if plot:
            valid_plot.set_data(valid_ticks, valid_score)
            train_plot.set_data(train_ticks, train_score)

            plt.axes().relim()
            plt.axes().autoscale_view(True, True, True)

            plt.pause(0.05)

    # Save trained svm.
    svm_pickle = "svm_%s.pkl" % time.strftime("%m_%d_%H_%M")
    with open(svm_pickle, 'wb') as fid:
        pickle.dump(svm, fid)
    print("Saved to %s." % svm_pickle)


def evaluate_using_k(fc2_layer, softmax_layer, train_group, valid_group,
        top_k=5, batch_size=8):
    # Normalize the activation feature vectors.
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)

    # SGDClassifier with hinge loss and l2 is an SVM.
    svm = linear_model.SGDClassifier(penalty='l2', alpha=0.0001, loss='hinge')

    for t, batch in enumerate(zip(train_group.generate(batch_size=batch_size),
                                  valid_group.generate(batch_size=batch_size))):
        if t >= NUM_BATCHES:
            break
        print('Batch #%d - %.2f%% completed.' % (t, t * 100.0 / NUM_BATCHES))

        # Images are mean centered and normalized.
        train, valid = batch

        # TODO: don't hardcode these.
        examples = np.zeros((batch_size, 2048))
        labels = np.zeros((batch_size, 10))

        valid_examples = np.zeros((batch_size, 2048))
        valid_labels = np.zeros((batch_size, 10))

        for i in range(batch_size):
            # Prep data for training.
            x_fc, x_sm, y = fc2_normal_entropy(train[0][i],
                    train[1][i], fc2_layer, fc2_normalize, softmax_layer)

            x_fc_valid, x_sm_valid, y_valid = fc2_normal_entropy(valid[0][i],
                    valid[1][i], fc2_layer, fc2_normalize, softmax_layer)

            print("Sorting.")
            x_sm.sort(key=lambda entropy_index: entropy_index[0])
            x_sm_valid.sort(key=lambda entropy_index: entropy_index[0])

            print("Element wise max")
            for k in range(top_k):
                for j in range(2048):
                    k_index = x_sm[k][1]

                    examples[i][j] = max(examples[i][j], x_fc[k_index][j])
                    valid_examples[i][j] = max(valid_examples[i][j],
                            x_fc_valid[k_index][j])

            print(y[0], y[1])
            labels[i] = y[0]
            valid_labels[i] = y_valid[0]

        # Train on batch.
        svm.partial_fit(examples, labels, classes=range(NUM_CLASSES))

        # Add accuracy to array.
        print("Train: %.4f" % svm.score(examples, labels))
        print("Valid: %.4f" % svm.score(valid_examples, valid_labels))



if __name__ == '__main__':
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    train_group = GroupedDatagen(TRAIN_DIR, preprocess=img_normalize)
    valid_group = GroupedDatagen(VALID_DIR, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model_vgg(SAVE_FILE)
    fc2_layer = extract_layer(model, 'fc2')
    softmax_layer = extract_layer(model, 'predictions')

    # Sort by top K minimized entropy.
    evaluate_using_k(fc2_layer, softmax_layer, train_group,
            valid_group, 3)

    if False:
        # Initialize data generators.
        train_datagen = get_data(TRAIN_DIR, BATCH_SIZE, preprocess=img_normalize)
        valid_datagen = get_data(VALID_DIR, BATCH_SIZE, preprocess=img_normalize)

        top_one_svm(fc2_layer, train_datagen, valid_datagen)
