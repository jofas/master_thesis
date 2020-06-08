from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np

from keras.layers import Dense as KDense
from keras.models import Sequential


N = 100


def test_inference():
    X = np.random.rand(500, N)

    kmodel = Sequential()
    kmodel.add(KDense(150, activation="relu", input_shape=(N,)))
    kmodel.add(KDense(150, activation="softmax"))
    kmodel.add(KDense(150, activation="tanh"))
    kmodel.add(KDense(150, activation="sigmoid"))
    kmodel.add(KDense(100, activation="softmax"))

    model = Model().add(Input(N)) \
                   .add(Dense(150, activation="relu")) \
                   .add(Dense(150, activation="softmax")) \
                   .add(Dense(150, activation="tanh")) \
                   .add(Dense(150, activation="sigmoid")) \
                   .add(Dense(100, activation="softmax"))

    model.set_weights_from_keras(kmodel.weights)

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 0.0001
