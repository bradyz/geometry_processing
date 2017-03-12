"""
Sample runs (from the package directory).

Fresh training:
    python3 scripts/generate_confusion_matrix.py \
            --k_features=5 \
            --svm_path=cache/svm_top_k_5_new.pkl
            --matrix_path=cache/svm_top_k_5
"""
import time
import argparse

import numpy as np

from sklearn.metrics import confusion_matrix

from geometry_processing.globals import (VALID_DIR, NUM_CLASSES,
        IMAGE_MEAN, IMAGE_STD, MODEL_WEIGHTS, FC2_MEAN, FC2_STD)

from geometry_processing.classification.multiview_model import MultiviewModel
from geometry_processing.utils.helpers import samplewise_normalize, extract_layer
from geometry_processing.train_cnn.classify_keras import load_model
from geometry_processing.utils.custom_datagen import GroupedDatagen


# Command line arguments.
parser = argparse.ArgumentParser(description='Generate a confusion matrix.')

parser.add_argument('--k_features', required=True, type=int,
        help='Number of features to consider.')
parser.add_argument('--svm_path', required=True, type=str,
        help='Path to the pickled SVM.')
parser.add_argument('--matrix_path', required=False, type=str, default='',
        help='Path (without extension) to save the matrix.')
parser.add_argument('--sort_mode', required=False, type=int, default=0,
        help='Scheme to pick top k (0 - greedy, 1 - random).')

args = parser.parse_args()
k_features = args.k_features
svm_path = args.svm_path
matrix_path = args.matrix_path
sort_mode = args.sort_mode


def evaluate_loop(mv_model, valid_group, batch_size=64, nb_epoch=100,
        save_file=None):
    # Start time.
    tic = time.time()

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for t, batch in enumerate(valid_group.generate(batch_size=batch_size)):
        if t >= nb_epoch:
            break
        # Batch start time.
        toc = time.time()

        x = batch[0]
        y_true = np.argmax(batch[1], axis=2)[:, 0]
        y_pred = mv_model.predict(x)

        matrix += confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

        print('Batch took %.4f seconds.' % (time.time() - toc))

    print('Total time elapsed - %.4f seconds.' % (time.time() - tic))
    print('Accuracy %.4f' % np.mean(np.diag(matrix) / np.sum(matrix, axis=1)))

    # Save matrix to disk.
    if save_file:
        print('Saving to %s.' % save_file)
        np.save(save_file, matrix)
        print('Success.')


if __name__ == '__main__':
    # Data source.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)
    valid_group = GroupedDatagen(VALID_DIR, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model(MODEL_WEIGHTS)
    fc2_layer = model.get_layer('fc2').output
    softmax_layer = model.get_layer('predictions').output

    # Training.
    multiview = MultiviewModel(model.layers[0].input, fc2_layer, softmax_layer,
            k_features, NUM_CLASSES, preprocess=fc2_normalize, svm_path=svm_path,
            sort_mode=sort_mode)

    evaluate_loop(multiview, valid_group, save_file=matrix_path)
