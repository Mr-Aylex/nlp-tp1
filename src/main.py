import pickle

import click
import numpy as np
from sklearn.model_selection import cross_val_score

from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model.main import make_model, dump_model
import logging

@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    logging.info("lunching train")
    df = make_dataset(input_filename)
    X, y = make_features(df, task, input_length=10)
    model = make_model(task)
    print("start training")
    model.fit(X, y)

    return None#dump_model(model, model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task, pos_tagger=True, remove_punctuation=True)

    with open(model_dump_filename, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    df["prediction"] = y_pred
    df.to_csv(output_filename, index=False)

    evaluate_model(model, X, y)

    return df


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model", default="models/model.pkl", help="File training model")
def evaluate(task, input_filename, model):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    with open(model, "rb") as f:
        model = pickle.load(f)

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {list(100 * scores)}"
          f" with mean {100 * np.mean(scores)}% ")

    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
