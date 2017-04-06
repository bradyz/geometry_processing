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
from geometry_processing.utils.helpers import samplewise_normalize
from geometry_processing.utils.custom_datagen import SaliencyDataGenerator


parser = argparse.ArgumentParser(description='Train a saliency NN.')

parser.add_argument('--verbose', required=False, type=int,
        default=1, help='[1] for curses, [2] for infrequent.')
parser.add_argument('--log_file', required=False, type=str,
        default='', help='File to log training, validation loss and accuracy.')

args = parser.parse_args()
verbose = args.verbose
log_file = args.log_file


def load_weights(model, weights_file):
    try:
        print('Loading weights from %s.' % weights_file)
        model.load_weights(weights_file, by_name=True)
    except OSError as e:
        print(e)
        print('Loading failed. Starting from scratch.')


def build_model(mvcnn_weights='', weights_file=''):
    img_input = Input(tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    base_model = VGG16(include_top=False, input_tensor=img_input)

    # Freeze all layers in pretrained network.
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='fc2')(x)

    # MVCNN.
    mvcnn = Model(input=img_input, output=x)

    if mvcnn_weights:
        load_weights(mvcnn, mvcnn_weights)

    # Freeze all layers in MVCNN.
    for layer in mvcnn.layers:
        layer.trainable = False

    full_x = mvcnn.output
    full_x = Dense(512, activation='relu',
            b_regularizer=regularizers.l2(0.01), name='fc3')(full_x)
    full_x = Dropout(0.5)(full_x)
    full_x = Dense(256, activation='relu',
            b_regularizer=regularizers.l2(0.01), name='fc4')(full_x)
    full_x = Dropout(0.5)(full_x)
    full_x = Dense(2, activation='softmax', name='predictions')(full_x)

    # Saliency predictor.
    saliency_model = Model(input=img_input, output=full_x)

    if weights_file:
        load_weights(saliency_model, weights_file)

    return saliency_model


def train(model, save_path):
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    # Various routines to run.
    callbacks = list()

    if log_file:
        callbacks.append(CSVLogger(log_file))

    if save_path:
        callbacks.append(ModelCheckpoint(filepath=save_path, verbose=1))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        patience=1, min_lr=0.0001))

    # Train.
    model.fit_generator(generator=train_generator.generate(),
            samples_per_epoch=train_generator.nb_data,
            nb_epoch=10,
            validation_data=valid_generator.generate(),
            nb_val_samples=1000,
            callbacks=callbacks,
            verbose=verbose)


if __name__ == '__main__':
    # Center and normalize each sample.
    normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    # Get streaming data.
    train_generator = SaliencyDataGenerator(TRAIN_DIR, SALIENCY_DATA_TRAIN,
            preprocess=normalize)
    valid_generator = SaliencyDataGenerator(VALID_DIR, SALIENCY_DATA_VALID,
            preprocess=normalize)

    print('%d training samples.' % train_generator.nb_data)
    print('%d validation samples.' % valid_generator.nb_data)

    saliency_cnn = build_model(MODEL_WEIGHTS, SALIENCY_MODEL)

    train(saliency_cnn, SALIENCY_MODEL)
