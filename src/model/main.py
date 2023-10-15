import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from src.model.dumb_model import MultinomialNBModel


def make_model():
    return Pipeline([
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", GradientBoostingClassifier()),
    ])


def dump_model(model, filename_output):
    pickle.dump(model, open(f"{filename_output}", "wb"))
