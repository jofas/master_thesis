from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np

from keras.layers import Dense as KDense
from keras.models import Sequential


def main():
    X = np.array(
        [[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32
    )

    y = np.array([0., 1., 1., 0.])

    kmodel = Sequential()
    kmodel.add(KDense(2, activation="relu", input_shape=(2,)))
    kmodel.add(KDense(1, activation="sigmoid"))

    model = Model().add(Input(2)) \
                   .add(Dense(2, activation="relu")) \
                   .add(Dense(1, activation="sigmoid"))

    model.set_weights_from_keras(kmodel.weights)

    p = model.predict(X)
    p_ = kmodel.predict(X, batch_size=4)

    assert np.all(p == p_)
    print("SUCCESS")


if __name__ == "__main__":
    main()
