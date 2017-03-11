import os

import numpy as np

from sklearn.metrics import confusion_matrix

from geometry_processing.globals import (VALID_DIR, NUM_CLASSES,
        IMAGE_MEAN, IMAGE_STD, MODEL_WEIGHTS, FC2_MEAN, FC2_STD, PACKAGE_PATH)

from geometry_processing.classification.multiview_model import MultiviewModel
from geometry_processing.utils.helpers import samplewise_normalize, extract_layer
from geometry_processing.train_cnn.classify_keras import load_model
from geometry_processing.utils.custom_datagen import GroupedDatagen


def evaluate(mv_model, valid_group, batch_size=32, nb_epoch=10, save_file=None):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for t, batch in enumerate(valid_group.generate(batch_size=batch_size)):
        if t >= nb_epoch:
            break

        x = batch[0]
        y_true = np.argmax(batch[1], axis=2)[:, 0]
        y_pred = mv_model.predict(x)

        matrix += confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    # Save matrix to disk.
    if save_file:
        np.save(save_file, matrix)


if __name__ == '__main__':
    # Data source.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)
    valid_group = GroupedDatagen(VALID_DIR, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model(MODEL_WEIGHTS)
    fc2_layer = extract_layer(model, 'fc2')
    softmax_layer = extract_layer(model, 'predictions')

    # Training.
    for i in [1, 3, 5, 7]:
        svm_path = os.path.join(PACKAGE_PATH, "cache", "svm_top_k_%d.pkl" % i)

        multiview = MultiviewModel(fc2_layer, softmax_layer, 3, NUM_CLASSES,
                preprocess=fc2_normalize, svm_path=svm_path)

        evaluate(multiview, valid_group)
