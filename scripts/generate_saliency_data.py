import keras.backend as K

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR,
        IMAGE_MEAN, IMAGE_STD, MODEL_WEIGHTS)

from geometry_processing.utils.helpers import samplewise_normalize, entropy
from geometry_processing.train_cnn.classify_keras import load_model
from geometry_processing.utils.custom_datagen import FilenameImageDatagen


def mean(subarray):
    return sum(subarray) / len(subarray)


def find_best_split(values):
    n = len(values)

    max_diff = float('-inf')
    max_i = 1

    for i in range(1, n):
        left = mean(values[:i])
        right = mean(values[i:])

        if max_diff < abs(right - left):
            max_diff = abs(right - left)
            max_i = i

    return max_i


if __name__ == '__main__':
    # Data source and image normalization.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)
    train_group = FilenameImageDatagen(TRAIN_DIR, preprocess=img_normalize)
    valid_group = FilenameImageDatagen(VALID_DIR, preprocess=img_normalize)

    # # Use the fc activations as features.
    model = load_model(MODEL_WEIGHTS)
    softmax_layer = model.get_layer('predictions').output

    # Wrapper around Tensorflow run operation.
    functor = K.function([model.layers[0].input, K.learning_phase()],
                         [softmax_layer])

    # Generate training data.
    for full_paths, images in train_group.generate():
        predictions = functor([images, 0.0])[0]

        n = predictions.shape[0]

        # Create a bunch of path, entropy score pairs.
        path_entropy = [(full_paths[i], entropy(predictions[i])) for i in range(n)]
        path_entropy.sort(key=lambda x: x[1])

        # Best index to partition the images.
        index_split = find_best_split([x[1] for x in path_entropy])

        # The more salient images.
        for i in range(index_split):
            print(path_entropy[i][0], 1.0)

        # The less salient images.
        for i in range(index_split, n):
            print(path_entropy[i][0], 0.0)
