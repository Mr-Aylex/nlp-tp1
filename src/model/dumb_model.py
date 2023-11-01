import pickle
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras import layers
import keras_nlp
import numpy as np
class DumbModel:
    def fit(self, X, y):
        print(self.model.summary())
        self.model.fit(X, y, epochs=15, batch_size=4, validation_split=0.2)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, filename_output):
        pickle.dump(self.model, open(f"{filename_output}", "wb"))

    def load(self, filename_input):
        self.model = pickle.load(open(filename_input, "rb"))

    def evaluate(self, X, y):
        return self.model.score(X, y)


class LSTMModel(DumbModel):
    """

    """

    def __init__(self, vocab_size, n_outputs=1, input_length=15, embed_dims=16, encoder_dims=16):
        self.model = keras.Sequential(
            [
                layers.Embedding(vocab_size, embed_dims, input_length=input_length),
                layers.LSTM(encoder_dims, return_sequences=True),
                layers.TimeDistributed(layers.Dense(n_outputs, activation='softmax') if n_outputs > 1 else layers.Dense(n_outputs, activation='sigmoid'))  # One output by sequence item


            ]
        )
        self.model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001), loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])

class GRUModel(DumbModel):
    """

    """
    def __init__(self, vocab_size, n_outputs=1, input_length=15, embed_dims=32, encoder_dims=32):
        self.model = keras.Sequential(
            [
                layers.Embedding(vocab_size, embed_dims, input_length=input_length),
                layers.GRU(encoder_dims, return_sequences=True),
                layers.TimeDistributed(layers.Dense(n_outputs, activation='softmax') if n_outputs > 1 else layers.Dense(n_outputs, activation='sigmoid')) # One output by sequence item
            ]
        )
        self.model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                           loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])
class TransformerModel(DumbModel):
    """

    """
    def __init__(self, vocab_size, n_outputs=1, input_length=15, embed_dims=32, encoder_dims=32):
        self.model = keras.Sequential(
            [
                layers.Embedding(vocab_size, embed_dims, input_length=input_length),
                keras_nlp.layers.TransformerEncoder(encoder_dims,4),
                keras_nlp.layers.TransformerDecoder(encoder_dims, 4),
                layers.TimeDistributed(layers.Dense(n_outputs, activation='softmax') if n_outputs > 1 else layers.Dense(n_outputs, activation='sigmoid')) # One output by sequence item
            ]
        )
        self.model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                           loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])


class TextVectoriser(DumbModel):
    def __init__(self, max_features, max_len):
        self.model = layers.TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=max_len)


    def fit(self, X, y):
        self.model.adapt(X)
        return self

    def transform(self, X):

        return self.model(X)