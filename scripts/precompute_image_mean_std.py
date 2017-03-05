import numpy as np

from geometry_processing.globals import TRAIN_DIR
from geometry_processing.utils.helpers import get_precomputed_statistics


mean, std = get_precomputed_statistics(TRAIN_DIR, 10000)

with open("mean.npy", "wb") as fd:
    np.save(fd, mean)
with open("std.npy", "wb") as fd:
    np.save(fd, std)
