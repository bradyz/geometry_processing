import pickle
import time

import numpy as np

from matplotlib import pyplot as plt

from sklearn import linear_model

from geometry_processing.train_cnn.classify_keras import load_model_vgg
from geometry_processing.globals import (TRAIN_DIR, VALID_DIR, NUM_CLASSES,
        IMAGE_MEAN, IMAGE_STD)
from geometry_processing.utils.helpers import (get_data, samplewise_normalize,
        extract_layer)

NUM_BATCHES = 10
BATCH_SIZE = 8


def top_one_svm(train_datagen, valid_datagen, plot=True):
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
        y = np.argmax(y, axis=1)

        x_valid, y_valid = valid_batch
        x_valid = fc2.predict(x_valid)
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


def evaluate_using_k(k=1):
    return


if __name__ == '__main__':
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Initialize data generators.
    train_datagen = get_data(TRAIN_DIR, BATCH_SIZE, preprocess=img_normalize)
    valid_datagen = get_data(VALID_DIR, BATCH_SIZE, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model_vgg()
    fc2 = extract_layer(model, 'fc2')

    # top_one_svm(train_datagen, valid_datagen)
