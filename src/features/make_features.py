import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

stop_words = set(stopwords.words('french'))
stemmer = FrenchStemmer()


def tokenize(x):
    return word_tokenize(x, language='french')


def delete_stopwords(x):
    return [i for i in x if not i in stop_words]


def stem(x):
    return [stemmer.stem(word) for word in x]


def re_join(x):
    return " ".join(x)


def tree_grams(x):
    return list(ngrams(x, 3))


def remove_punctuation(x):
    return [word for word in x if word.isalpha()]


def make_features(df, task):
    y = get_output(df, task)

    df['video_name'] = df['video_name'].apply(lambda x: tokenize(x))
    df['video_name'] = df['video_name'].apply(lambda x: remove_punctuation(x))
    df['video_name'] = df['video_name'].apply(lambda x: delete_stopwords(x))
    df['video_name'] = df['video_name'].apply(lambda x: stem(x))
    df['video_name'] = df['video_name'].apply(lambda x: ' '.join(x))

    X = df["video_name"]
    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
