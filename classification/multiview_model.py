import pickle

import numpy as np

from sklearn import linear_model


def entropy(x):
    return -np.sum(x * np.log(x))


class MultiviewModel:
    def __init__(self, feature_layer, softmax_layer, k, svm_path=None):
        self.feature_layer = feature_layer
        self.softmax_layer = softmax_layer
        self.k = k

        if svm_path:
            with open(svm_path, "rb") as fd:
                self.svm = pickle.load(fd)
        else:
            self.svm = linear_model.SGDClassifier(penalty='l2', alpha=0.0002,
                    loss='hinge')

    def get_top_k(self, x):
        n = x.shape[0]

        features = self.feature_layer.predict(x)
        softmax = self.softmax_layer.predict(x)

        entropy_index = [(entropy(softmax[i]), i) for i in range(n)]
        entropy_index.sort(key=lambda x: x[0])

        # Create a chunk using only the most confident views.
        top_k_features = np.zeros((self.k, features.shape[1]))

        # Grab the top k.
        for i in range(self.k):
            ith_index = entropy_index[i][1]
            top_k_features[i] = features[ith_index]

        return top_k_features

    def predict(self, batch, preprocess=None, postprocess=None):
        batch_size = batch.shape[0]

        if preprocess is not None:
            batch = np.apply_along_axis(preprocess, 1, batch)

        examples = np.zeros((batch_size, 2048))

        for i in range(batch_size):
            x = batch[i]
            examples[i] = self.get_top_k(x)

        if postprocess is not None:
            examples = np.apply_along_axis(postprocess, 1, examples)

        return self.svm.predict(examples)
