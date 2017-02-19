from keras.applications.vgg16 import VGG16
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


TRAIN_DIR = '/home/04365/bradyz/data/ModelNetViewpoints/train/'
VALID_DIR = '/home/04365/bradyz/data/ModelNetViewpoints/test/'
SAVE_FILE = 'model_weights.h5'
LOG_FILE = 'training.log'
NUM_CLASSES = 10
IMAGE_SIZE = 224
TRAIN = 99775
VALID = 5000


def train(model):
    train_datagen = ImageDataGenerator(samplewise_center=True,
            samplewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=64)

    valid_datagen = ImageDataGenerator(samplewise_center=True,
            samplewise_std_normalization=True)

    valid_generator = valid_datagen.flow_from_directory(
            VALID_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=64)

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=SAVE_FILE, verbose=1)
    csv_logger = CSVLogger(LOG_FILE)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=0.001)

    model.fit_generator(generator=train_generator,
            samples_per_epoch=TRAIN,
            nb_epoch=10,
            validation_data=valid_generator,
            nb_val_samples=VALID,
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
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(input=img_input, output=x)
    # model.load_weights(SAVE_FILE, by_name=True)

    return model


if __name__ == "__main__":
    model = load_model_vgg()

    train(model)
