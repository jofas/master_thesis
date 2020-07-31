from spiDNN import Model
from spiDNN.layers import Input, Dense, Conv1D

import numpy as np

from keras.layers import Dense as KDense, Conv1D as KConv1D, Flatten
from keras.models import Sequential


N = 224
n_channels = 3
kernel_size = 3


def test_inference():
    X = np.random.rand(500, N)

    kmodel = Sequential()
    kmodel.add(KDense(50, activation="relu", input_shape=(N,)))
    kmodel.add(KDense(50, activation="softmax"))
    kmodel.add(KDense(300, activation="tanh"))
    kmodel.add(KDense(50, activation="sigmoid"))
    kmodel.add(KDense(25, activation="softmax"))

    model = Model().add(Input(N)) \
                   .add(Dense(50, activation="relu")) \
                   .add(Dense(50, activation="softmax")) \
                   .add(Dense(300, activation="tanh")) \
                   .add(Dense(50, activation="sigmoid")) \
                   .add(Dense(25, activation="softmax"))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 1e-4


def test_inference_conv1d():
    input_shape = (N, n_channels)
    X = np.random.rand(500, *input_shape)

    kmodel = Sequential()
    kmodel.add(KConv1D(1, 7, padding="same", input_shape=input_shape, strides=2, activation="relu"))
    kmodel.add(KConv1D(5, kernel_size * 4, padding="same", strides=3, activation="tanh"))
    kmodel.add(KConv1D(16, kernel_size, padding="same", strides=2, activation="sigmoid"))
    kmodel.add(KConv1D(16, kernel_size + 3, padding="same", strides=5, activation="softmax"))
    kmodel.add(KConv1D(5, kernel_size + 1, strides=3))
    kmodel.add(Flatten())
    kmodel.add(KDense(1))

    model = Model()
    model.add(Input(*input_shape))
    model.add(Conv1D(1, (7,), padding="same", stride=2, activation="relu"))
    model.add(Conv1D(5, (kernel_size * 4,), padding="same", stride=3, activation="tanh"))
    model.add(Conv1D(16, (kernel_size,), padding="same", stride=2, activation="sigmoid"))
    model.add(Conv1D(16, (kernel_size + 3,), padding="same", stride=5, activation="softmax"))
    model.add(Conv1D(5, (kernel_size + 1,), stride=3))
    model.add(Dense(1))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 1e-4


if __name__ == "__main__":
    # test_inference()
    test_inference_conv1d()
    print("SUCCESS.")
