import argparse

import numpy as np

from geometry_processing.globals import VALID_DIR
from geometry_processing.utils.helpers import plot_confusion_matrix, get_data


# Command line arguments.
parser = argparse.ArgumentParser(description='View a confusion matrix.')

parser.add_argument('--matrix_path', required=True, type=str,
        help='Path to the pickled (via numpy) matrix.')

args = parser.parse_args()
matrix_path = args.matrix_path


def get_class_labels(data_generator):
    # Something like ['bathtub', 'desk', ...].
    class_labels = data_generator.class_indices
    class_labels = list(sorted(class_labels, key=lambda x: class_labels[x]))
    return class_labels


if __name__ == '__main__':
    data_generator = get_data(VALID_DIR)

    matrix = np.load(matrix_path)
    plot_confusion_matrix(matrix, get_class_labels(data_generator),
            normalize=True, title="Confusion")
