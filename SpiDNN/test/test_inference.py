from spiDNN import Model
from spiDNN.layers import Input, Dense, Conv1D

import numpy as np

from keras.layers import Dense as KDense, Conv1D as KConv1D, \
    Flatten as KFlatten
from keras.models import Sequential


N = 100
n_channels = 1
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

# TODO:
#       3. then more channels (must be implemented into off-board io
#       4. then more filters, etc.
#
# channel counter -> array, makes it more secure/independent from
#                    other neurons
def test_inference_conv1d():
    X = np.random.rand(500, N, n_channels)

    kmodel = Sequential()
    kmodel.add(KConv1D(
        1, kernel_size, input_shape=(N, n_channels), padding="same"))
    kmodel.add(KFlatten())
    kmodel.add(KDense(1, activation=None))

    model = Model().add(Input(N)) \
                   .add(Conv1D(
                       (kernel_size,), "identity", padding="same")) \
                   .add(Dense(1, activation="identity"))

    model.set_weights(kmodel.get_weights())

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 0.0001


if __name__ == "__main__":
    #test_inference()
    test_inference_conv1d()
    print("SUCCESS.")
