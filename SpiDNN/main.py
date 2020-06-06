from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np

from keras.layers import Dense as KDense
from keras.models import Sequential

N = 5  # problem is not time but something else. 5 is max


def main():
    X = np.random.rand(100, N)

    """
    X = np.array(
        [[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32
    )
    """

    # y = np.array([0., 1., 1., 0.])

    kmodel = Sequential()
    kmodel.add(KDense(150, activation="relu", input_shape=(N,)))
    kmodel.add(KDense(150, activation="softmax"))
    kmodel.add(KDense(150, activation="tanh"))
    kmodel.add(KDense(150, activation="sigmoid"))
    kmodel.add(KDense(150, activation="softmax"))

    model = Model().add(Input(N)) \
                   .add(Dense(150, activation="relu")) \
                   .add(Dense(150, activation="softmax")) \
                   .add(Dense(150, activation="tanh")) \
                   .add(Dense(150, activation="sigmoid")) \
                   .add(Dense(150, activation="softmax")) \

    model.set_weights_from_keras(kmodel.weights)

    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    assert np.amax(error) < 0.0001
    print("SUCCESS")


if __name__ == "__main__":
    main()
