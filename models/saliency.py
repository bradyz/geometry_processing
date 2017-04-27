import numpy as np

from sklearn.metrics import confusion_matrix

from keras import regularizers
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR,
        SALIENCY_DATA_TRAIN, SALIENCY_DATA_VALID,
        IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_CLASSES)
from geometry_processing.utils.helpers import samplewise_normalize
from geometry_processing.utils.custom_datagen import SaliencyDataGenerator
from geometry_processing.models import multiview_cnn


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


def train(model, save_path, nb_epoch=10, nb_val_samples=1000, batch_size=64,
        log_file=None, verbose=1):
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


def test(model, batch_size=32):
    # Optimizer is unused.
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])

    # Center and normalize each sample.
    normalize = samplewise_normalize(IMAGE_MEAN, IMAGE_STD)

    test_generator = SaliencyDataGenerator(VALID_DIR, SALIENCY_DATA_VALID,
            preprocess=normalize, batch_size=batch_size)

    print('%d validation samples.' % test_generator.nb_data)

    # Either right or wrong.
    matrix = np.zeros((2, 2))

    for x, y_true in test_generator.generate():
        if test_generator.epochs_seen == 1:
            break

        # Convert probabilities to predictions.
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(model.predict_on_batch(x), axis=1)

        matrix += confusion_matrix(y_true, y_pred, labels=range(2))

    return matrix
