import numpy as np
import tensorflow as tf

from spiDNN import Model
from spiDNN.layers import Input, Dense, Conv1D


N = 224
n_channels = 3
kernel_size = 3


def test_inference():
    X = np.random.rand(500, N)

    kmodel = tf.keras.models.Sequential()
    kmodel.add(tf.keras.layers.Dense(50, activation="relu", input_shape=(N,)))
    kmodel.add(tf.keras.layers.Dense(50, activation="softmax"))
    kmodel.add(tf.keras.layers.Dense(300, activation="tanh"))
    kmodel.add(tf.keras.layers.Dense(50, activation="sigmoid"))
    kmodel.add(tf.keras.layers.Dense(25))
    kmodel.add(tf.keras.layers.Dense(17, activation="softmax"))

    model = Model()
    model.add(Input(N))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="softmax"))
    model.add(Dense(300, activation="tanh"))
    model.add(Dense(50, activation="sigmoid"))
    model.add(Dense(25))
    model.add(Dense(17, activation="softmax"))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 1e-4


def test_inference_conv1d_1():
    input_shape = (N, n_channels)
    X = np.random.rand(500, *input_shape)

    kmodel = tf.keras.models.Sequential()
    kmodel.add(tf.keras.layers.Conv1D(
        1, kernel_size, padding="same", input_shape=input_shape))
    kmodel.add(tf.keras.layers.Conv1D(5, kernel_size * 4, padding="same"))
    kmodel.add(tf.keras.layers.Conv1D(
        16, kernel_size, padding="same", strides=2))
    kmodel.add(tf.keras.layers.Flatten())
    kmodel.add(tf.keras.layers.Dense(16))

    model = Model()
    model.add(Input(*input_shape))
    model.add(Conv1D(1, (kernel_size,), padding="same"))
    model.add(Conv1D(5, (kernel_size * 4,), padding="same"))
    model.add(Conv1D(16, (kernel_size,), padding="same", stride=2))
    model.add(Dense(16))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 1e-4


def test_inference_conv1d_2():
    input_shape = (N, n_channels)
    X = np.random.rand(500, *input_shape)

    kmodel = tf.keras.models.Sequential()
    kmodel.add(tf.keras.layers.Conv1D(
        1, 7, padding="same", input_shape=input_shape, strides=2,
        activation="relu"))
    kmodel.add(tf.keras.layers.Conv1D(
        5, kernel_size * 4, padding="same", strides=3, activation="tanh"))
    kmodel.add(tf.keras.layers.Conv1D(
        16, kernel_size, padding="same", strides=2, activation="sigmoid"))
    kmodel.add(tf.keras.layers.Conv1D(
        16, kernel_size + 3, padding="same", strides=5, activation="softmax"))
    kmodel.add(tf.keras.layers.Conv1D(5, kernel_size + 1, strides=3))
    kmodel.add(tf.keras.layers.Flatten())
    kmodel.add(tf.keras.layers.Dense(1))

    model = Model()
    model.add(Input(*input_shape))
    model.add(Conv1D(1, (7,), padding="same", stride=2, activation="relu"))
    model.add(Conv1D(
        5, (kernel_size * 4,), padding="same", stride=3, activation="tanh"))
    model.add(Conv1D(
        16, (kernel_size,), padding="same", stride=2, activation="sigmoid"))
    model.add(Conv1D(
        16, (kernel_size + 3,), padding="same", stride=5,
        activation="softmax"))
    model.add(Conv1D(5, (kernel_size + 1,), stride=3))
    model.add(Dense(1))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 1e-4


if __name__ == "__main__":
    # test_inference()
    test_inference_conv1d_1()
    # test_inference_conv1d_2()
    print("SUCCESS.")
