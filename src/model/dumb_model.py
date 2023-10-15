import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


class DumbModel:
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, filename_output):
        pickle.dump(self.model, open(f"{filename_output}", "wb"))

    def load(self, filename_input):
        self.model = pickle.load(open(filename_input, "rb"))

    def evaluate(self, X, y):
        return self.model.score(X, y)


class MultinomialNBModel(DumbModel):
    """

    """

    def __init__(self):
        self.model = MultinomialNB()
