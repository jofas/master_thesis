from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np

import tensorflow as tf

from keras.layers import Dense as KDense
from keras.models import Sequential
from keras.optimizers import SGD

from copy import deepcopy
from time import time

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.1

def test_mlp_backprop():
    X = np.array([[.0, .0], [.0, 1.], [1., .0], [1., 1.]])
    y = np.array([[.0, 1.], [1., .0], [1., .0], [.0, 1.]])

    kmodel = tf.keras.models.Sequential()
    kmodel.add(tf.keras.layers.Dense(
        50, activation="relu", input_shape=(2,)))
    kmodel.add(tf.keras.layers.Dense(50, activation="softmax"))
    kmodel.add(tf.keras.layers.Dense(300, activation="tanh"))
    kmodel.add(tf.keras.layers.Dense(50, activation="sigmoid"))
    kmodel.add(tf.keras.layers.Dense(25))
    kmodel.add(tf.keras.layers.Dense(2, activation="softmax"))

    kmodel.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE))

    model=Model()
    model.add(Input(2))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="softmax"))
    model.add(Dense(300, activation="tanh"))
    model.add(Dense(50, activation="sigmoid"))
    model.add(Dense(25))
    model.add(Dense(2, activation="softmax"))

    model.set_weights(kmodel.get_weights())

    unfitted_weights = deepcopy(model.get_weights())

    spinn_start = time()
    model.fit(
        X, y, "mean_squared_error", epochs=EPOCHS, batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE)
    spinn_end = time()

    keras_start = time()
    kmodel.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)
    keras_end = time()

    w = model.get_weights()
    w_ = kmodel.get_weights()

    error = [x - x_ for x, x_ in zip(w, w_)]
    update = [x - x_ for x, x_ in zip(w, unfitted_weights)]

    # make sure weights are updated in SDRAM before they are extracted
    for u in update:
        assert np.amax(np.absolute(u)) > 0.0

    for e in error:
        e_max = np.amax(np.absolute(e))
        print(e_max)
        assert e_max < 0.1

    print("SpiNNaker took: {} seconds".format(spinn_end - spinn_start))
    print("Keras took: {} seconds".format(keras_end - keras_start))


def compare_against_keras(X, y, loss):
    kmodel = Sequential()
    kmodel.add(KDense(64, activation="relu", input_shape=(X.shape[1],)))
    kmodel.add(KDense(64, activation="tanh"))
    kmodel.add(KDense(64, activation="softmax"))
    kmodel.add(KDense(y.shape[1], activation="sigmoid"))

    kmodel.compile(loss=loss, optimizer=SGD(learning_rate=LEARNING_RATE))

    model = Model().add(Input(X.shape[1])) \
                   .add(Dense(64, activation="relu")) \
                   .add(Dense(64, activation="tanh")) \
                   .add(Dense(64, activation="softmax")) \
                   .add(Dense(y.shape[1], activation="sigmoid"))

    model.set_weights(kmodel.get_weights())

    unfitted_weights = deepcopy(model.get_weights())

    model.fit(
        X, y, loss, epochs=EPOCHS, batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE)

    kmodel.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)

    w = model.get_weights()
    w_ = kmodel.get_weights()

    error = [x - x_ for x, x_ in zip(w, w_)]
    update = [x - x_ for x, x_ in zip(w, unfitted_weights)]

    for u in update:
        assert np.amax(np.absolute(u)) > 0.0

    for e in error:
        e_max = np.amax(np.absolute(e))
        print(e_max)
        assert e_max < 0.1

    """
    p = model.predict(X)
    p_ = kmodel.predict(X)

    e_max = np.amax(np.absolute(p - p_))
    print(e_max)
    assert e_max < 0.1
    """


def test_binary_xor():
    loss = "binary_crossentropy"

    X = np.array([[.0, .0], [.0, 1.], [1., .0], [1., 1.]])
    y = np.array([[.0], [1.], [1.], [.0]])

    compare_against_keras(X, y, loss)


def test_categorical_xor():
    loss = "categorical_crossentropy"
    loss = "mean_squared_error"

    X = np.array([[.0, .0], [.0, 1.], [1., .0], [1., 1.]])
    y = np.array([[1., .0], [.0, 1.], [.0, 1.], [1., .0]])

    compare_against_keras(X, y, loss)


if __name__ == "__main__":
    # test_binary_xor()
    # test_categorical_xor()
    test_mlp_backprop()
    print("SUCCESS.")
