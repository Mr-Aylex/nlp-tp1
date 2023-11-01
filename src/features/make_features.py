import logging
import ast
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.tag import StanfordPOSTagger
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
from collections import Counter
from tqdm import tqdm
tqdm.pandas()
stop_words = set(stopwords.words('french'))
stemmer = FrenchStemmer()
stanford_dir = "data/pos_tagger/stanford-postagger-full-2020-11-17/"
modelfile = stanford_dir + "models/french-ud.tagger"
jarfile = stanford_dir + "stanford-postagger-4.2.0.jar"
tagger = StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)

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


def app_remove_punctuation(x):
    return [word for word in x if word.isalpha()]

def app_pos_tagger(x):
    tagged = tagger.tag(x)
    tagged = Counter(tag for word, tag in tagged)
    return tagged

def make_features(df, task, **kwargs):
    print("make feature")
    y = get_output(df, task, **kwargs)
    X = get_inputs(df, **kwargs)

    return X, y


def normaliser_taille(arr, taille_cible):
    longueur_actuelle = len(arr)
    if longueur_actuelle > taille_cible:
        # Couper le tableau si plus long que la taille cible
        return arr[:taille_cible]
    elif longueur_actuelle < taille_cible:
        # Étendre le tableau avec des zéros (ou toute autre valeur) si plus court
        return np.pad(arr, (0, taille_cible - longueur_actuelle), 'constant')
    else:
        return arr
def get_output(df, task, **kwargs):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"].apply(ast.literal_eval).apply(lambda x: np.array(x, dtype=np.float32))

        y = y.apply(lambda x: normaliser_taille(x, kwargs["input_length"]))
        y = np.stack(y.values)
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y



def get_inputs(df, **kwargs):

    if "remove_stopwords" not in kwargs:
        remove_stopwords = False
    else:
        remove_stopwords = kwargs["remove_stopwords"]

    if "stemming" not in kwargs:
        stemming = False
    else:
        stemming = kwargs["stemming"]

    if "ngrams" not in kwargs:
        ngrams = False
    else:
        ngrams = kwargs["ngrams"]

    if "pos_tagger" not in kwargs:
        pos_tagger = False
    else:
        pos_tagger = kwargs["pos_tagger"]

    if "remove_punctuation" not in kwargs:
        remove_punctuation = False
    else:
        remove_punctuation = kwargs["remove_punctuation"]


    if ngrams and pos_tagger:
        logging.error("you can't apply ngrams with pos_tagger")
        raise Exception
    logging.info("start preprocessing")
    df['video_name'] = df['video_name'].apply(lambda x: tokenize(x))

    if remove_punctuation:
        logging.info("starting remove punctuation")
        df['video_name'] = df['video_name'].progress_apply(lambda x: app_remove_punctuation(x))

    if remove_stopwords:
        logging.info("starting delete stopwords")
        df['video_name'] = df['video_name'].progress_apply(lambda x: delete_stopwords(x))
    if stemming:
        logging.info("starting stem")
        df['video_name'] = df['video_name'].progress_apply(lambda x: stem(x))
    if ngrams:
        logging.info("starting tree grams")
        df['video_name'] = df['video_name'].progress_apply(lambda x: tree_grams(x))
    if pos_tagger:
        logging.info("starting pos_tagger")
        df['video_name'] = df['video_name'].progress_apply(lambda x: app_pos_tagger(x))
    if not pos_tagger:

        df['video_name'] = df['video_name'].progress_apply(lambda x: re_join(x))

    x = df['video_name']
    return x
