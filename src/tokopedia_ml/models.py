"""Keras model builders used in the Tokopedia notebook."""

from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_lstm(
    vocab: int,
    edim: int,
    mlen: int,
    nc: int,
    units: int = 128,
    drop: float = 0.3,
    bidir: bool = False,
    stacked: bool = False,
) -> keras.Model:
    """Build an LSTM or BiLSTM classifier.

    Args:
        vocab: vocabulary size (embedding input_dim).
        edim: embedding dimension.
        mlen: maximum sequence length.
        nc: number of output classes.
        units: LSTM units.
        drop: dropout rate for final dense layer.
        bidir: whether to use Bidirectional LSTM.
        stacked: whether to stack a second LSTM layer.
    """
    inp = keras.Input(shape=(mlen,))
    x = layers.Embedding(vocab, edim, mask_zero=True)(inp)
    x = layers.SpatialDropout1D(0.2)(x)

    def rnn(u, rs):
        cell = layers.LSTM(u, return_sequences=rs)
        return layers.Bidirectional(cell) if bidir else cell

    x = rnn(units, stacked)(x)
    if stacked:
        x = rnn(units // 2, False)(x)
    else:
        x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(drop)(x)
    out = layers.Dense(nc, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_textcnn(
    vocab: int,
    edim: int,
    mlen: int,
    nc: int,
    filters: int = 128,
    kernels: tuple = (2, 3, 4, 5),
    drop: float = 0.3,
) -> keras.Model:
    """Build a Kim-2014 style TextCNN classifier."""
    inp = keras.Input(shape=(mlen,))
    emb = layers.Embedding(vocab, edim)(inp)
    emb = layers.SpatialDropout1D(0.2)(emb)
    pools = [
        layers.GlobalMaxPooling1D()(
            layers.Conv1D(filters, k, activation="relu", padding="same")(emb)
        )
        for k in kernels
    ]
    x = layers.concatenate(pools)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(
        x
    )
    x = layers.Dropout(drop)(x)
    out = layers.Dense(nc, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
