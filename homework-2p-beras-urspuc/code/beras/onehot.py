import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    TODO:
        - fit
        - call
        - inverse

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        ## TODO: Fetch all the unique labels and create a dictionary with
        ## the unique labels as keys and their one hot encodings as values

        self.numbers = np.unique(data)
        self.vecs = np.eye(len(self.numbers))
        self.onehot = {e: self.vecs[i] for i, e in enumerate(self.numbers)}
        self.inverse_mapping = {tuple(self.vecs[i]): e for i, e in enumerate(self.numbers)}

    def call(self, data):
        ## TODO: Implement call function
        if not hasattr(self, 'onehot'):
            self.fit(data)
        return np.array([self.onehot[x] for x in data])

    def inverse(self, data):
        ## TODO: Implement inverse function
        if not hasattr(self, 'inverse_mapping'):
            raise Exception("Encoder not fitted!")
        return np.array([self.inverse_mapping[tuple(x)] for x in data])