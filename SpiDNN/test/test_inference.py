from spiDNN import Model
from spiDNN.layers import Input, Dense, Conv1D

import numpy as np

from keras.layers import Dense as KDense, Conv1D as KConv1D, \
    Flatten as KFlatten
from keras.models import Sequential


N = 100
n_channels = 4
kernel_size = 8


def test_inference():
    X = np.random.rand(500, N)

    kmodel = Sequential()
    kmodel.add(KDense(100, activation="relu", input_shape=(N,)))
    kmodel.add(KDense(100, activation="softmax"))
    kmodel.add(KDense(300, activation="tanh"))
    kmodel.add(KDense(100, activation="sigmoid"))
    kmodel.add(KDense(100, activation="softmax"))

    model = Model().add(Input(N)) \
                   .add(Dense(100, activation="relu")) \
                   .add(Dense(100, activation="softmax")) \
                   .add(Dense(300, activation="tanh")) \
                   .add(Dense(100, activation="sigmoid")) \
                   .add(Dense(100, activation="softmax"))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 0.0001


def test_inference_conv1d():
    input_shape = (N, n_channels)
    X = np.random.rand(500, *input_shape)

    kmodel = Sequential()
    kmodel.add(KConv1D(
        8, kernel_size * 4, input_shape=input_shape, padding="same"))
    kmodel.add(KConv1D(16, kernel_size * 2, padding="same"))
    kmodel.add(KConv1D(1, kernel_size, padding="same"))
    kmodel.add(KFlatten())
    kmodel.add(KDense(1, activation=None))

    model = Model().add(Input(*input_shape)) \
                   .add(Conv1D(8, (kernel_size * 4,), padding="same")) \
                   .add(Conv1D(16, (kernel_size * 2,), padding="same")) \
                   .add(Conv1D(1, (kernel_size,), padding="same")) \
                   .add(Dense(1, activation="identity"))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 1e-4


if __name__ == "__main__":
    #test_inference()
    test_inference_conv1d()
    print("SUCCESS.")
