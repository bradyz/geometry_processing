from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR, MODEL_WEIGHTS,
        LOG_FILE, IMAGE_SIZE, NUM_CLASSES, IMAGE_MEAN, IMAGE_STD)
from geometry_processing.utils.helpers import (get_data,
        get_precomputed_statistics, samplewise_normalize, load_weights)


# Set to 2 when training on supercomputer (one line per epoch).
VERBOSITY = 2
USE_SAVE = True


def train(model, save_to=''):
    # Center and normalize each sample.
    normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Get streaming data.
    train_generator = get_data(TRAIN_DIR, preprocess=normalize)
    valid_generator = get_data(VALID_DIR, preprocess=normalize)

    print('%d training samples.' % train_generator.n)
    print('%d validation samples.' % valid_generator.n)

    optimizer = SGD(lr=0.01, momentum=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    callbacks = list()

    callbacks.append(CSVLogger(LOG_FILE))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        patience=2, min_lr=0.0001))

    if save_to:
        callbacks.append(ModelCheckpoint(filepath=MODEL_WEIGHTS, verbose=1))

    model.fit_generator(generator=train_generator,
            samples_per_epoch=train_generator.n,
            nb_epoch=5,
            validation_data=valid_generator,
            nb_val_samples=1000,
            callbacks=callbacks,
            verbose=VERBOSITY)

    # Save the weights on completion.
    if save_to:
        model.save_weights(save_to)


def load_model(input_tensor=None, include_top=True):
    if input_tensor is None:
        input_tensor = Input(tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

    # Don't include VGG fc layers.
    base_model = VGG16(include_top=False, input_tensor=input_tensor)

    # Freeze all layers in pretrained network.
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)

    if include_top:
        x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    return Model(input=input_tensor, output=x)


def test(model, nb_batch=32, nb_worker=4):
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=[])

    # Center and normalize each sample.
    normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Get streaming data.
    test_generator = get_data(VALID_DIR, preprocess=normalize)

    return model.evaluate_generator(test_generator, nb_batch, nb_worker=nb_worker)


if __name__ == '__main__':
    mvcnn = load_model()
    load_weights(mvcnn, MODEL_WEIGHTS)

    # train(mvcnn, save_to=MODEL_WEIGHTS)
    print("Test loss: %.3f" % test(mvcnn))
