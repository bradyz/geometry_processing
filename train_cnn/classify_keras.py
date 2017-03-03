from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Model

from geometry_processing.globals import (TRAIN_DIR, VALID_DIR, SAVE_FILE,
        LOG_FILE, IMAGE_SIZE, NUM_CLASSES)
from geometry_processing.utils.helpers import get_data


USE_SAVE = True


def train(model):
    train_generator = get_data(TRAIN_DIR)
    valid_generator = get_data(VALID_DIR)

    print("%d training samples." % train_generator.n)
    print("%d validation samples." % valid_generator.n)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=SAVE_FILE, verbose=1)
    csv_logger = CSVLogger(LOG_FILE)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.001)

    model.fit_generator(generator=train_generator,
            samples_per_epoch=train_generator.n,
            nb_epoch=10,
            validation_data=valid_generator,
            nb_val_samples=valid_generator.n,
            callbacks=[checkpointer, csv_logger, reduce_lr])

    model.save_weights(SAVE_FILE)


def load_model_vgg():
    img_input = Input(tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

    base_model = VGG16(include_top=False, input_tensor=img_input)

    # Freeze all layers in pretrained network.
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.1)(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(0.1)(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(input=img_input, output=x)

    if USE_SAVE:
        print('Loading weights from %s.' % SAVE_FILE)
        model.load_weights(SAVE_FILE, by_name=True)

    return model


if __name__ == "__main__":
    model = load_model_vgg()
    train(model)
