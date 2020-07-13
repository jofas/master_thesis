from spiDNN import Model
from spiDNN.layers import Input, Dense, Conv1D

import numpy as np

from keras.layers import Dense as KDense, Conv1D as KConv1D, \
    Flatten as KFlatten
from keras.models import Sequential


N = 200
n_channels = 4
kernel_size = 3


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
    assert np.amax(error) < 1e-4


def test_inference_conv1d():
    input_shape = (N, n_channels)
    X = np.random.rand(500, *input_shape)

    kmodel = Sequential()
    kmodel.add(KConv1D(1, kernel_size, input_shape=input_shape, strides=1))
    kmodel.add(KConv1D(1, kernel_size * 4, padding="same", strides=3))
    kmodel.add(KConv1D(1, kernel_size, padding="same", strides=2))
    kmodel.add(KConv1D(1, kernel_size +3, padding="same", strides=5))
    kmodel.add(KConv1D(5, kernel_size, strides=1))
    kmodel.add(KFlatten())
    kmodel.add(KDense(1, activation=None))

    # TODO: weird EIEIO input crap when first layer "valid" and stride > 1
    #
    #       "valid" padding may leaves last neuron of prev layer unconnected
    #       which results in an error (catch that)
    #
    model = Model()
    model.add(Input(*input_shape))
    model.add(Conv1D(1, (kernel_size,), stride=1))
    model.add(Conv1D(1, (kernel_size * 4,), padding="same", stride=3))
    model.add(Conv1D(1, (kernel_size,), padding="same", stride=2))
    model.add(Conv1D(1, (kernel_size +3,), padding="same", stride=5))
    model.add(Conv1D(5, (kernel_size,), stride=1))
    model.add(Dense(1, activation="identity"))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 1e-4


if __name__ == "__main__":
    # test_inference()
    test_inference_conv1d()
    print("SUCCESS.")
