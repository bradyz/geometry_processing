import os
import time
import numpy as np

import geometry_processing


PACKAGE_PATH = os.path.dirname(geometry_processing.__file__)

IMAGE_SIZE = 224
BATCH = 64
NUM_CLASSES = 10

IMAGE_MEAN = np.load(os.path.join(PACKAGE_PATH, "cache/image_mean.npy"))
IMAGE_STD = np.load(os.path.join(PACKAGE_PATH, "cache/image_std.npy"))

FC2_MEAN = np.load(os.path.join(PACKAGE_PATH, "cache/fc2_mean.npy"))
FC2_STD = np.load(os.path.join(PACKAGE_PATH, "cache/fc2_std.npy"))

MODEL_WEIGHTS = os.path.join(PACKAGE_PATH, "cache", "model_weights_V2.h5")
SALIENCY_MODEL = os.path.join(PACKAGE_PATH, "cache", "saliency_model.h5")

LOG_FILE = os.path.join(PACKAGE_PATH, "logs",
                        "training_%s.log" % time.strftime("%m_%d_%H_%M"))

SALIENCY_DATA_TRAIN = os.path.join(PACKAGE_PATH, "data", "saliency_labels_train.txt")
SALIENCY_DATA_VALID = os.path.join(PACKAGE_PATH, "data", "saliency_labels_valid.txt")

################################################################################
# Machine Specific Paths.
################################################################################
# Local mac machine paths.
TRAIN_DIR = "/Users/bradyzhou/code/data/ModelNetViewpoints/train/"
VALID_DIR = "/Users/bradyzhou/code/data/ModelNetViewpoints/test/"

# Local unix machine paths.
# TRAIN_DIR = "/home/brady/code/data/ModelNetViewpoints/train/"
# VALID_DIR = "/home/brady/code/data/ModelNetViewpoints/test/"

# TACC supercomputer information.
# TRAIN_DIR = "/home/04365/bradyz/data/ModelNetViewpoints/train/"
# VALID_DIR = "/home/04365/bradyz/data/ModelNetViewpoints/test/"
################################################################################
