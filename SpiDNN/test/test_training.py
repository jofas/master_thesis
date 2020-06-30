from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np

from keras.layers import Dense as KDense
from keras.models import Sequential
from keras.optimizers import SGD

from copy import deepcopy

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.1

def test_training():
    X = np.array([[.0, .0], [.0, 1.], [1., .0], [1., 1.]])
    y = np.array([[.0], [1.], [1.], [.0]])

    kmodel = Sequential()
    #kmodel.add(KDense(16, activation="tanh", input_shape=(2,)))
    #kmodel.add(KDense(16, activation="relu"))
    kmodel.add(KDense(3, activation="tanh", input_shape=(2,)))
    #kmodel.add(KDense(3, activation="sigmoid"))
    kmodel.add(KDense(1, activation="sigmoid"))

    kmodel.compile(loss="mean_squared_error", optimizer=SGD(
        learning_rate=LEARNING_RATE))

    model = Model().add(Input(2)) \
                   .add(Dense(3, activation="tanh")) \
                   .add(Dense(1, activation="sigmoid"))
                   #.add(Dense(16, activation="tanh")) \
                   #.add(Dense(16, activation="relu")) \
                   #.add(Dense(3, activation="sigmoid")) \

    model.set_weights(kmodel.get_weights())

    unfitted_weights = deepcopy(model.get_weights())

    model.fit(X, y, "mean_squared_error", epochs=EPOCHS,
              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    kmodel.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)

    w = model.get_weights()
    w_ = kmodel.get_weights()

    error = [x - x_ for x, x_ in zip(w, w_)]
    update = [x - x_ for x, x_ in zip(w, unfitted_weights)]

    #print(unfitted_weights)
    #print(w)
    #print(w_)
    #print(error)

    for u in update:
        assert np.amax(np.absolute(u)) > 0.0
    for e in error:
        assert np.amax(np.absolute(e)) < 0.075

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
