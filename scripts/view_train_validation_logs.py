import os
import csv

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline

from geometry_processing.globals import PACKAGE_PATH


EPOCHS = 50
SMOOTH = 10


def show_graph(k_train, k_valid, show_train=False):
    plt.ylim([0, 1])

    if show_train:
        for k, train in k_train:
            plt.plot(train, label='train score: %d' % k)

    for k, valid in k_valid:
        plt.plot(valid, label='valid score: %d' % k)

    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    k_train = list()
    k_valid = list()

    for k in [1, 3, 10, 15]:
        csv_path = os.path.join(PACKAGE_PATH, "logs", "top_%d.log" % k)

        batch = list()
        train = list()
        valid = list()

        with open(csv_path) as fd:
            for i, row in enumerate(csv.DictReader(fd)):
                if i >= EPOCHS:
                    break
                batch.append(row['batch'])
                train.append(row['train_acc'])
                valid.append(row['val_acc'])

        batch = np.array(batch, dtype=np.float32)
        train = np.array(train, dtype=np.float32)
        valid = np.array(valid, dtype=np.float32)

        batch_smooth = np.linspace(batch.min(), batch.max(), SMOOTH)

        train_smooth = spline(batch, train, batch_smooth)
        valid_smooth = spline(batch, valid, batch_smooth)

        k_train.append((k, train_smooth))
        k_valid.append((k, valid_smooth))

    show_graph(k_train, k_valid)
