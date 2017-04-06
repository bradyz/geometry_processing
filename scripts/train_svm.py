"""
Sample runs (from the package directory).

Fresh training:
    python3 scripts/train_svm.py \
            --k_features=5 \
            --save_path=cache/svm_top_k_5_new.pkl

Fresh training, using random sort:
    python3 scripts/train_svm.py \
            --k_features=5 \
            --save_path=cache/svm_top_k_5_new.pkl \
            --sort_mode=1

Continued training from a previous run:
    python3 scripts/train_svm.py \
            --k_features=5 \
            --svm_path=cache/svm_top_k_5.pkl \
            --save_path=cache/svm_top_k_5_new.pkl
"""
import time
import argparse

import numpy as np

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR, NUM_CLASSES,
        IMAGE_MEAN, IMAGE_STD, MODEL_WEIGHTS, FC2_MEAN, FC2_STD)
from geometry_processing.utils.helpers import samplewise_normalize
from geometry_processing.utils.custom_datagen import GroupedDatagen
from geometry_processing.models.multiview_cnn import load_model
from geometry_processing.models.multiview_svm import MultiviewModel


# Command line arguments.
parser = argparse.ArgumentParser(description='Train a top K SVM.')

parser.add_argument('--k_features', required=True, type=int,
        help='Number of features to consider.')
parser.add_argument('--svm_path', required=False, type=str, default='',
        help='Path to the pre-trained SVM (leave blank if fresh start).')
parser.add_argument('--save_path', required=False, type=str, default='',
        help='Path to save the trained SVM.')
parser.add_argument('--sort_mode', required=False, type=int, default=0,
        help='Scheme to pick top k (0 - greedy, 1 - random).')

args = parser.parse_args()
k_features = args.k_features
svm_path = args.svm_path
save_path = args.save_path
sort_mode = args.sort_mode


def train_loop(mv_model, train_group, valid_group, batch=64, nb_batches=50,
        save_file=None):
    # Start time.
    tic = time.time()

    for t, batch in enumerate(zip(train_group.generate(batch_size=batch),
                                  valid_group.generate(batch_size=batch // 4))):
        # Have seen enough batches. Terminate.
        if t >= nb_batches:
            break
        # Batch start time.
        toc = time.time()

        # Both train and valid are tuples.
        train, valid = batch

        # Sample train: (32, 25, 224, 224, 3), (32, 25, 10).
        train_x = train[0]
        train_y = np.argmax(train[1], axis=2)[:, 0]

        valid_x = valid[0]
        valid_y = np.argmax(valid[1], axis=2)[:, 0]

        score = mv_model.fit(train_x, train_y)

        print('Training accuracy %.4f' % score)
        print('Validation accuracy %.4f' % mv_model.score(valid_x, valid_y))
        print('Batch training took %.4f seconds.' % (time.time() - toc))

    print('Total time elapsed - %.4f seconds.' % (time.time() - tic))

    # Keep the model's weights.
    if save_file:
        mv_model.save(save_file)


if __name__ == '__main__':
    # Data source.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)
    train_group = GroupedDatagen(TRAIN_DIR, preprocess=img_normalize)
    valid_group = GroupedDatagen(VALID_DIR, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model(MODEL_WEIGHTS)
    fc2_layer = model.get_layer('fc2').output
    softmax_layer = model.get_layer('predictions').output

    # Training.
    multiview = MultiviewModel(model.layers[0].input, fc2_layer, softmax_layer,
            k_features, NUM_CLASSES, preprocess=fc2_normalize, svm_path=svm_path,
            sort_mode=sort_mode)

    train_loop(multiview, train_group, valid_group, save_file=save_path)
