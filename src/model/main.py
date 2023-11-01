import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from src.model.dumb_model import LSTMModel, TextVectoriser, TransformerModel
import logging

def make_model(task):
    logging.info("create model")
    if task == "is_comic_video":
        return Pipeline([
            ("count_vectorizer", CountVectorizer()),
            ("random_forest", GradientBoostingClassifier()),
        ])
    elif task == "is_name":
        return Pipeline([
            ("vectorizer", TextVectoriser(max_features=1000, max_len=10)),
            ('classifier', TransformerModel(vocab_size=2000, input_length=10))
        ])
    elif task == "find_comic_name":
        pass



def dump_model(model, filename_output):
    pickle.dump(model, open(f"{filename_output}", "wb"))
