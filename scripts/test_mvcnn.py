import argparse

import numpy as np

from geometry_processing.globals import MODEL_WEIGHTS
from geometry_processing.utils.helpers import load_weights
from geometry_processing.models.multiview_cnn import load_model, test


parser = argparse.ArgumentParser(description='Test MVCNN classification.')

parser.add_argument('--matrix_path', required=True, type=str,
        help='Path to save the confusion matrix.')

args = parser.parse_args()
matrix_path = args.matrix_path


if __name__ == '__main__':
    # Initialize model.
    mvcnn = load_model()
    load_weights(mvcnn, MODEL_WEIGHTS)

    # Run through entire test dataset.
    matrix = test(mvcnn)
    print('Accuracy %.4f' % np.mean(np.diag(matrix) / np.sum(matrix, axis=1)))

    # Save matrix to disk.
    if matrix_path:
        print('Saving to %s.' % matrix_path)
        np.save(matrix_path, matrix)
