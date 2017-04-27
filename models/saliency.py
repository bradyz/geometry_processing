import argparse

from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR, MODEL_WEIGHTS,
        SALIENCY_DATA_TRAIN, SALIENCY_DATA_VALID, SALIENCY_MODEL,
        IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD)
from geometry_processing.utils.helpers import (samplewise_normalize,
        load_weights)
from geometry_processing.utils.custom_datagen import SaliencyDataGenerator
from geometry_processing.models import multiview_cnn


parser = argparse.ArgumentParser(description='Train a saliency NN.')

parser.add_argument('--verbose', required=False, type=int,
        default=1, help='[1] for ncurses, [2] for per epoch.')
parser.add_argument('--log_file', required=False, type=str,
        default='', help='File to log training, validation loss and accuracy.')

args = parser.parse_args()
verbose = args.verbose
log_file = args.log_file


def build_model(input_tensor=None):
    if input_tensor is None:
        input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    mvcnn = multiview_cnn.load_model(input_tensor=input_tensor, include_top=False)

    x = mvcnn.output
    x = Dense(256, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01), name='fc3')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            bias_regularizer=regularizers.l2(0.01), name='fc4')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax', name='saliency')(x)

    # Saliency predictor.
    return Model(inputs=input_tensor, outputs=x)


def train(model, save_path, nb_epoch=10, nb_val_samples=1000, batch_size=64):
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    # Center and normalize each sample.
    normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Get streaming data.
    train_generator = SaliencyDataGenerator(TRAIN_DIR, SALIENCY_DATA_TRAIN,
            preprocess=normalize, batch_size=batch_size)
    valid_generator = SaliencyDataGenerator(VALID_DIR, SALIENCY_DATA_VALID,
            preprocess=normalize, batch_size=batch_size)

    print('%d training samples.' % train_generator.nb_data)
    print('%d validation samples.' % valid_generator.nb_data)

    # Various routines to run.
    callbacks = list()

    if log_file:
        callbacks.append(CSVLogger(log_file))

    if save_path:
        callbacks.append(ModelCheckpoint(filepath=save_path, verbose=1))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
        patience=0, min_lr=1e-5))

    # Train.
    model.fit_generator(generator=train_generator.generate(),
            steps_per_epoch=train_generator.nb_data // batch_size,
            epochs=nb_epoch,
            validation_data=valid_generator.generate(),
            validation_steps=nb_val_samples // batch_size,
            callbacks=callbacks,
            verbose=verbose)


if __name__ == '__main__':
    # Build and load cached weights.
    saliency_cnn = build_model()
    load_weights(saliency_cnn, MODEL_WEIGHTS)

    # Update model.
    train(saliency_cnn, save_path=SALIENCY_MODEL)
