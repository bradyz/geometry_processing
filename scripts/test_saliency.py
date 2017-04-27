import argparse

import numpy as np

from geometry_processing.globals import SALIENCY_MODEL
from geometry_processing.utils.helpers import load_weights
from geometry_processing.models.saliency import build_model, test


parser = argparse.ArgumentParser(description='Test saliency classification.')

parser.add_argument('--matrix_path', required=True, type=str,
        help='Path to save the confusion matrix.')

args = parser.parse_args()
matrix_path = args.matrix_path


if __name__ == '__main__':
    # Initialize model.
    saliency_cnn = build_model()
    load_weights(saliency_cnn, SALIENCY_MODEL)

    # Run through entire test dataset.
    matrix = test(saliency_cnn)
    print('Per Class Accuracy %s' % (np.diag(matrix) / np.sum(matrix, axis=1)))
    print('MAP: %.4f' % np.mean(np.diag(matrix) / np.sum(matrix, axis=1)))

    # Save matrix to disk.
    if matrix_path:
        print('Saving to %s.' % matrix_path)
        np.save(matrix_path, matrix)
