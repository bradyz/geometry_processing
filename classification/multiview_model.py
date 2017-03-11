import pickle

import numpy as np

from sklearn import linear_model


def entropy(x):
    return -np.sum(x * np.log(x))


class MultiviewModel:
    def __init__(self, feature_layer, softmax_layer, k, nb_classes,
            svm_path=None, preprocess=None):
        self.feature_layer = feature_layer
        self.softmax_layer = softmax_layer
        self.k = k
        self.nb_classes = nb_classes

        self.preprocess = preprocess

        if svm_path:
            print('Loading SVM from %s.' % svm_path)
            with open(svm_path, "rb") as fd:
                self.svm = pickle.load(fd)
        else:
            self.svm = linear_model.SGDClassifier(penalty='l2', alpha=0.0002,
                    loss='hinge')

    def get_top_k_features(self, x):
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

    def aggregated_features(self, batch):
        batch_size = batch.shape[0]

        examples = np.zeros((batch_size, 2048))

        for i in range(batch_size):
            top_k_features = self.get_top_k_features(batch[i])
            examples[i] = np.max(top_k_features, axis=0)

        if self.preprocess is not None:
            examples = np.apply_along_axis(self.preprocess, 1, examples)

        return examples

    def fit(self, x, y):
        x_ = self.aggregated_features(x)
        self.svm.partial_fit(x_, y, classes=range(self.nb_classes))

    def predict(self, batch):
        return self.svm.predict(self.aggregated_features(batch))

    def score(self, x, y):
        return self.score(self.aggregated_features(x), y)

    def save(self, file_path):
        print('Saving to %s.' % file_path)
        with open(file_path, 'wb') as fd:
            pickle.dump(self.svm, fd)
        print('Success.')
