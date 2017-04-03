import keras.backend as K

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR,
        IMAGE_MEAN, IMAGE_STD, MODEL_WEIGHTS)

from geometry_processing.utils.helpers import samplewise_normalize, entropy
from geometry_processing.train_cnn.classify_keras import load_model
from geometry_processing.utils.custom_datagen import FilenameImageDatagen


if __name__ == '__main__':
    # Data source and image normalization.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)
    train_group = FilenameImageDatagen(TRAIN_DIR, preprocess=img_normalize)
    valid_group = FilenameImageDatagen(VALID_DIR, preprocess=img_normalize)

    # # Use the fc activations as features.
    model = load_model(MODEL_WEIGHTS)
    fc2_layer = model.get_layer('fc2').output
    softmax_layer = model.get_layer('predictions').output

    functor = K.function([model.layers[0].input, K.learning_phase()],
                         [softmax_layer])

    for full_paths, images in train_group.generate():
        probabilities = functor([images, 0.0])[0]

        for i in range(probabilities.shape[0]):
            print(probabilities[i])
            print(entropy(probabilities[i]))
