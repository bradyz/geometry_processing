import numpy as np

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR, NUM_CLASSES,
        IMAGE_MEAN, IMAGE_STD, SAVE_FILE, FC2_MEAN, FC2_STD)

from geometry_processing.classification.multiview_model import MultiviewModel
from geometry_processing.utils.helpers import samplewise_normalize, extract_layer
from geometry_processing.train_cnn.classify_keras import load_model_vgg
from geometry_processing.utils.custom_datagen import GroupedDatagen


def train(mv_model, train_group, valid_group, batch_size=32):
    for t, batch in enumerate(zip(train_group.generate(batch_size=batch_size),
                                  valid_group.generate(batch_size=batch_size))):
        train, valid = batch

        # Sample train[0]: (32, 25, 224, 224, 3).
        # Sample train[1]: (32, 25, 10).
        train_x = train[0]
        train_y = np.argmax(train[1], axis=2)[:, 0]

        mv_model.fit(train_x, train_y)


if __name__ == '__main__':
    # Data source.
    img_normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)
    fc2_normalize = samplewise_normalize(FC2_MEAN, FC2_STD)
    train_group = GroupedDatagen(TRAIN_DIR, preprocess=img_normalize)
    valid_group = GroupedDatagen(VALID_DIR, preprocess=img_normalize)

    # Use the fc activations as features.
    model = load_model_vgg(SAVE_FILE)
    fc2_layer = extract_layer(model, 'fc2')
    softmax_layer = extract_layer(model, 'predictions')

    # Training.
    multiview = MultiviewModel(fc2_layer, softmax_layer, 3, NUM_CLASSES,
            preprocess=fc2_normalize)
    train(multiview, train_group, valid_group)
