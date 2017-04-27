import argparse

import keras.backend as K

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR,
        IMAGE_MEAN, IMAGE_STD, MODEL_WEIGHTS)
from geometry_processing.utils.helpers import (samplewise_normalize,
        entropy, load_weights)
from geometry_processing.utils.custom_datagen import FilenameImageDatagen
from geometry_processing.models.multiview_cnn import load_model


parser = argparse.ArgumentParser(description='Generate saliency data.')

parser.add_argument('--confidence_threshold', required=False, type=float,
        default=0.7, help='Value from [0, inf]. Lower is more confident.')
parser.add_argument('--pick_top', required=False, type=int,
        default=-1, help='Value from [-1, num_views].')
parser.add_argument('--generate_dataset', required=True, type=str,
        help='train/test')

args = parser.parse_args()
confidence_threshold = args.confidence_threshold
pick_top = args.pick_top
generate_dataset = args.generate_dataset


# Set pick_top to -1 if thresholding by entropy value.
def generate(datagen, functor):
    # Generate training data. 25 to ensure viewpoints are batched (hack).
    for full_paths, images in datagen.generate(25):
        # 0 means test mode (turn off dropout).
        predictions = functor([images, 0])[0]

        if pick_top > -1:
            paths_predictions = list(zip(full_paths, predictions))
            paths_predictions.sort(key=lambda x: entropy(x[1]))

            for i, path_prediction in enumerate(paths_predictions):
                if i < pick_top:
                    print(path_prediction[0], 1.0)
                else:
                    print(path_prediction[0], 0.0)
        else:
            for i in range(predictions.shape[0]):
                if entropy(predictions[i]) <= confidence_threshold:
                    print(full_paths[i], 1.0)
                else:
                    print(full_paths[i], 0.0)


if __name__ == '__main__':
    # Data source and image normalization.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Directory of images.
    if generate_dataset == 'train':
        datagen = FilenameImageDatagen(TRAIN_DIR, preprocess=img_normalize)
    elif generate_dataset == 'test':
        datagen = FilenameImageDatagen(VALID_DIR, preprocess=img_normalize)
    else:
        raise ValueError('Invalid input for --generate_dataset.')

    # # Use the fc activations as features.
    model = load_model()
    load_weights(model, MODEL_WEIGHTS)

    # Wrapper around Tensorflow run operation.
    functor = K.function([model.layers[0].input, K.learning_phase()],
                         [model.get_layer('predictions').output])

    generate(datagen, functor)
