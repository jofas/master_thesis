from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np

from keras.layers import Dense as KDense
from keras.models import Sequential


def test_training():
    X = np.array([[.0, .0], [.0, 1.], [1., .0], [1., 1.]])
    y = np.array([[.0], [1.], [1.], [.0]])

    kmodel = Sequential()
    kmodel.add(KDense(2, activation="sigmoid", input_shape=(2,)))
    kmodel.add(KDense(1, activation="sigmoid"))
    kmodel.compile(loss="mean_squared_error", optimizer="sgd")

    model = Model().add(Input(2)) \
                   .add(Dense(2, activation="sigmoid")) \
                   .add(Dense(1, activation="sigmoid"))

    model.set_weights_from_keras(kmodel.weights)

    print(model.get_weights())

    kmodel.fit(X, y, epochs=1, batch_size=4, shuffle=False)
    model.fit(X, y, "mean_squared_error", epochs=1, batch_size=4)

    w = model.get_weights()
    w_ = kmodel.get_weights()

    error = [x - x_ for x, x_ in zip(w, w_)]

    print(error)

    """
    p = model.predict(X)
    p_ = kmodel.predict(X)

    error = np.absolute(p - p_)
    print(error)
    #assert np.amax(error) < 0.0002
    """


if __name__ == "__main__":
    test_training()
    print("SUCCESS.")
