import argparse

from geometry_processing.globals import SALIENCY_MODEL, MODEL_WEIGHTS
from geometry_processing.utils.helpers import load_weights
from geometry_processing.models.saliency import build_model, train


parser = argparse.ArgumentParser(description='Train a saliency NN.')

parser.add_argument('--verbose', required=False, type=int,
        default=1, help='[1] for ncurses, [2] for per epoch.')
parser.add_argument('--log_file', required=False, type=str,
        default='', help='File to log training, validation loss and accuracy.')

args = parser.parse_args()
verbose = args.verbose
log_file = args.log_file


if __name__ == '__main__':
    # Build and load cached weights.
    saliency_cnn = build_model()
    load_weights(saliency_cnn, MODEL_WEIGHTS)

    # Update model.
    train(saliency_cnn, save_path=SALIENCY_MODEL)
