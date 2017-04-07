import numpy as np
from matplotlib import pyplot as plt

from geometry_processing.globals import (VALID_DIR, MODEL_WEIGHTS,
        SALIENCY_DATA_VALID, SALIENCY_MODEL, IMAGE_MEAN, IMAGE_STD)
from geometry_processing.utils.helpers import samplewise_normalize
from geometry_processing.utils.custom_datagen import SaliencyDataGenerator
from geometry_processing.models.saliency import build_model


TO_SHOW = 3
GRID_FORMAT = '{}{}%s'.format(TO_SHOW, TO_SHOW)


def denormalize(img):
    return np.rint(np.clip(img * IMAGE_STD + IMAGE_MEAN, 0.0, 255.0))


def run(model, datagen):
    plt.figure(1)

    for imgs, labels in datagen.generate():
        prediction = model.predict(imgs)

        for i in range(TO_SHOW):
            for j in range(TO_SHOW):
                index = i * TO_SHOW + j

                plt.subplot(GRID_FORMAT % (index + 1))
                plt.title('%.3f/%.3f ' % (prediction[index][1], labels[index][1]))
                plt.imshow(255.0 - denormalize(imgs[index]))

        plt.show()


if __name__ == '__main__':
    # Center and normalize each sample.
    normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Get streaming data.
    valid_generator = SaliencyDataGenerator(VALID_DIR, SALIENCY_DATA_VALID,
            preprocess=normalize, batch_size=TO_SHOW ** 2)

    print('%d samples.' % valid_generator.nb_data)

    saliency_cnn = build_model(MODEL_WEIGHTS, SALIENCY_MODEL)

    run(saliency_cnn, valid_generator)
