from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np

from keras.layers import Dense as KDense
from keras.models import Sequential


K = 3


def test_training():
    X = np.random.rand(500, K)
    y = np.random.rand(500, K)

    """
    kmodel = Sequential()
    kmodel.add(KDense(150, activation="relu", input_shape=(N,)))
    kmodel.add(KDense(150, activation="softmax"))
    kmodel.add(KDense(150, activation="tanh"))
    kmodel.add(KDense(150, activation="sigmoid"))
    kmodel.add(KDense(100, activation="softmax"))
    """

    model = Model().add(Input(K)) \
                   .add(Dense(2, activation="relu")) \
                   .add(Dense(2, activation="softmax")) \
                   .add(Dense(2, activation="tanh")) \
                   .add(Dense(2, activation="sigmoid")) \
                   .add(Dense(K, activation="softmax"))

    model.fit(X, y, "mean_squared_error", epochs=5, batch_size=1024)

    """
    model.set_weights_from_keras(kmodel.weights)

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 0.0001
    """


if __name__ == "__main__":
    test_training()
    print("SUCCESS.")
