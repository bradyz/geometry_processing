import argparse

import keras.backend as K

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR,
        IMAGE_MEAN, IMAGE_STD, MODEL_WEIGHTS)

from geometry_processing.utils.helpers import samplewise_normalize, entropy
from geometry_processing.train_cnn.classify_keras import load_model
from geometry_processing.utils.custom_datagen import FilenameImageDatagen


parser = argparse.ArgumentParser(description='Generate saliency data.')

parser.add_argument('--confidence_threshold', required=False, type=float,
        default=0.7, help='Value from [0, inf]. Lower is more confident.')
parser.add_argument('--generate_training', required=True, type=int,
        help='[1] for training, [2] for validation.')

args = parser.parse_args()
confidence_threshold = args.confidence_threshold
generate_training = args.generate_training


def generate(datagen, functor):
    # Generate training data.
    for full_paths, images in datagen.generate(25):
        predictions = functor([images, 0.0])[0]

        for i in range(predictions.shape[0]):
            if entropy(predictions[i]) <= confidence_threshold:
                print(full_paths[i], 1.0)
            else:
                print(full_paths[i], 0.0)


if __name__ == '__main__':
    # Data source and image normalization.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Directory of images.
    if generate_training == 1:
        datagen = FilenameImageDatagen(TRAIN_DIR, preprocess=img_normalize)
    elif generate_training == 2:
        datagen = FilenameImageDatagen(VALID_DIR, preprocess=img_normalize)

    # # Use the fc activations as features.
    model = load_model(MODEL_WEIGHTS)
    softmax_layer = model.get_layer('predictions').output

    # Wrapper around Tensorflow run operation.
    functor = K.function([model.layers[0].input, K.learning_phase()],
                         [softmax_layer])

    generate(datagen, functor)
