import numpy as np

from geometry_processing.globals import (TRAIN_DIR, IMAGE_MEAN,
        IMAGE_STD, SAVE_FILE)
from geometry_processing.train_cnn.classify_keras import load_model_vgg
from geometry_processing.utils.helpers import (get_data, samplewise_normalize,
        extract_layer)


# Number of samples to use for std and mean calculations.
NUM_SAMPLES = 45000


def get_mean_std(layer, datagen, num_samples):
    samples = np.zeros((num_samples, layer.output_shape[1]))

    index = 0
    for x, _ in datagen:
        if index >= num_samples:
            break
        print(index)

        activations = layer.predict(x)

        offset = min(num_samples - index, activations.shape[0])
        samples[index:index+offset] = activations[:offset]
        index += offset

    print(samples[-1])

    return np.mean(samples, axis=0), np.std(samples, axis=0)


if __name__ == '__main__':
    # Use the fc activations as features.
    model = load_model_vgg(SAVE_FILE)
    fc2 = extract_layer(model, 'fc2')

    # Normalize the image.
    normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Initialize data generators.
    train_datagen = get_data(TRAIN_DIR, 64, preprocess=normalize)

    # Precompute fc2 center and std.
    mean, std = get_mean_std(fc2, train_datagen, NUM_SAMPLES)

    # Cache for future use.
    with open("fc2_mean.npy", "wb") as fd:
        np.save(fd, mean)
    with open("fc2_std.npy", "wb") as fd:
        np.save(fd, std)

    print(mean, std)
