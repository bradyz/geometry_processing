import numpy as np

from matplotlib import pyplot as plt

from sklearn import linear_model

from keras.models import Model

from geometry_processing.train_cnn.classify_keras import load_model_vgg
from geometry_processing.globals import TRAIN_DIR, VALID_DIR, NUM_CLASSES
from geometry_processing.utils.helpers import get_data


NUM_BATCHES = 100
BATCH_SIZE = 16


def extract_layer(full_model, layer='fc2'):
    intermediate_layer_model = Model(input=full_model.input,
                                     output=full_model.get_layer(layer).output)
    return intermediate_layer_model


if __name__ == '__main__':
    train_datagen = get_data(TRAIN_DIR, BATCH_SIZE)
    valid_datagen = get_data(VALID_DIR, BATCH_SIZE)

    model = load_model_vgg()
    fc2 = extract_layer(model)

    svm = linear_model.SGDClassifier(penalty='l2', alpha=0.001,
                                     loss='hinge', average=True)

    valid_score = list()
    valid_ticks = list()

    train_score = list()
    train_ticks = list()

    plt.ion()
    plt.xlabel('Batch')
    plt.ylabel('Score')

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_autoscale_on(True)

    train_plot = plt.plot(train_ticks, train_score, label='train score')[0]
    valid_plot = plt.plot(valid_ticks, valid_score, label='valid score')[0]
    plt.legend(loc='best')
    plt.show()

    for i, batch in enumerate(zip(train_datagen, valid_datagen)):
        if i >= NUM_BATCHES:
            break
        print(i)

        train_batch, valid_batch = batch

        x, y = train_batch
        x = fc2.predict(x)
        y = np.argmax(y, axis=1)

        x_valid, y_valid = valid_batch
        x_valid = fc2.predict(x_valid)
        y_valid = np.argmax(y_valid, axis=1)

        svm.partial_fit(x, y, classes=range(NUM_CLASSES))

        train_ticks.append(i)
        train_score.append(svm.score(x, y))

        valid_ticks.append(i)
        valid_score.append(svm.score(x_valid, y_valid))

        valid_plot.set_data(valid_ticks, valid_score)
        train_plot.set_data(train_ticks, train_score)

        axes.relim()
        axes.autoscale_view(True, True, True)

        plt.pause(0.05)

    while True:
        plt.pause(0.05)

    import pdb; pdb.set_trace()
