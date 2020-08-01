from spiDNN import Model
from spiDNN.layers import Input, Dense, Conv1D

import numpy as np

from keras.layers import Dense as KDense, Conv1D as KConv1D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

from copy import deepcopy
import time

EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.1


def compare_against_keras(X, y, loss):
    kmodel = Sequential()
    kmodel.add(KDense(64, activation="relu", input_shape=(X.shape[1],)))
    kmodel.add(KDense(64, activation="tanh"))
    kmodel.add(KDense(64, activation="softmax"))
    kmodel.add(KDense(y.shape[1], activation="sigmoid"))

    kmodel.compile(loss=loss, optimizer=SGD(learning_rate=LEARNING_RATE))

    model = Model()
    model.add(Input(X.shape[1]))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="tanh"))
    model.add(Dense(64, activation="softmax"))
    model.add(Dense(y.shape[1], activation="sigmoid"))

    model.set_weights(kmodel.get_weights())

    unfitted_weights = deepcopy(model.get_weights())

    model.fit(
        X, y, loss, epochs=EPOCHS, batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE)

    kmodel.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)

    w = model.get_weights()
    w_ = kmodel.get_weights()

    error = [x - x_ for x, x_ in zip(w, w_)]
    update = [x - x_ for x, x_ in zip(w, unfitted_weights)]

    for u in update:
        assert np.amax(np.absolute(u)) > 0.0

    for e in error:
        e_max = np.amax(np.absolute(e))
        print(e_max)
        assert e_max < 0.2

    """
    p = model.predict(X)
    p_ = kmodel.predict(X)

    e_max = np.amax(np.absolute(p - p_))
    print(e_max)
    assert e_max < 0.1
    """


def test_binary_xor():
    loss = "binary_crossentropy"

    X = np.array([[.0, .0], [.0, 1.], [1., .0], [1., 1.]])
    y = np.array([[.0], [1.], [1.], [.0]])

    compare_against_keras(X, y, loss)


def test_categorical_xor():
    loss = "categorical_crossentropy"
    loss = "mean_squared_error"

    X = np.array([[.0, .0], [.0, 1.], [1., .0], [1., 1.]])
    y = np.array([[1., .0], [.0, 1.], [.0, 1.], [1., .0]])

    compare_against_keras(X, y, loss)


def test_training_conv1d():
    loss = "mean_squared_error"
    kernel_size = 3
    input_shape = (50, 3)

    X = np.random.rand(500, *input_shape)
    y = np.random.rand(500, 4)

    kmodel = Sequential()
    kmodel.add(KConv1D(1, 3, padding="same", input_shape=input_shape))
    kmodel.add(KConv1D(1, kernel_size * 4, padding="same", strides=1, activation="tanh"))
    kmodel.add(KConv1D(1, kernel_size, padding="same", strides=1, activation="sigmoid"))
    #kmodel.add(KConv1D(16, kernel_size + 3, padding="same", activation="softmax"))
    #kmodel.add(KConv1D(5, kernel_size + 1))
    kmodel.add(Flatten())
    kmodel.add(KDense(y.shape[1]))

    kmodel.compile(loss=loss, optimizer=SGD(learning_rate=LEARNING_RATE))

    model = Model()
    model.add(Input(*input_shape))
    model.add(Conv1D(1, (3,), padding="same", activation="relu"))
    model.add(Conv1D(1, (kernel_size * 4,), padding="same", stride=1, activation="tanh"))
    model.add(Conv1D(1, (kernel_size,), padding="same", stride=1, activation="sigmoid"))
    #model.add(Conv1D(16, (kernel_size + 3,), padding="same", activation="softmax"))
    #model.add(Conv1D(5, (kernel_size + 1,)))
    model.add(Dense(y.shape[1]))

    model.set_weights(kmodel.get_weights())

    unfitted_weights = deepcopy(model.get_weights())

    model.fit(
        X, y, loss, epochs=3, batch_size=256,
        learning_rate=LEARNING_RATE)

    kmodel.fit(X, y, epochs=3, batch_size=256, shuffle=False)

    w = model.get_weights()
    w_ = kmodel.get_weights()

    error = [x - x_ for x, x_ in zip(w, w_)]
    update = [x - x_ for x, x_ in zip(w, unfitted_weights)]

    for u in update:
        assert np.amax(np.absolute(u)) > 0.0

    print(w)
    print(w_)

    for e in error:
        e_max = np.amax(np.absolute(e))
        print(e_max)
        #assert e_max < 0.1


def test_training_conv1d_with_known_weights():
    input_shape = (2, 1)

    X = np.array([[0., 1.]]).reshape(1,2,1)
    y = np.array([[1.]])

    weights = [
        np.array([[[.1, .4]],
                  [[.2, .5]],
                  [[.3, .6]]]), np.array([.0, .0]),
        np.array([[[ .7, 1.3],
                   [ .8, 1.4]],
                  [[ .9, 1.5],
                   [1. , 1.6]],
                  [[1.1, 1.7],
                   [1.2, 1.8]]]), np.array([.0, .0]),
        np.array([[1.9],[2.0],[2.1],[2.2]]), np.array([.0])
    ]


    c1 = KConv1D(2, 3, padding="same", input_shape=input_shape)
    c2 = KConv1D(2, 3, padding="same")

    kmodel = Sequential()
    kmodel.add(c1)
    kmodel.add(c2)
    kmodel.add(Flatten())
    kmodel.add(KDense(1))

    kmodel.compile(loss="mean_squared_error", optimizer=SGD(learning_rate=1.0))

    kmodel.set_weights(weights)

    kmodel.train_on_batch(X, y)

    model = Model()
    model.add(Input(*input_shape))
    model.add(Conv1D(2, (3,), padding="same"))
    model.add(Conv1D(2, (3,), padding="same"))
    model.add(Dense(1))

    model.set_weights(deepcopy(weights))

    #model.predict(X)

    #print(model._layers[2].neurons[0].weights)
    #print(len(model._layers[2].neurons[0].weights))

    model.fit(X, y, "mean_squared_error", epochs=1, batch_size=1,
              learning_rate=1.0)

    w = model.get_weights()
    w_ = kmodel.get_weights()

    error = [x - x_ for x, x_ in zip(w, w_)]
    update = [x - x_ for x, x_ in zip(w, weights)]

    for u in update:
        assert np.amax(np.absolute(u)) > 0.0

    for e in error:
        e_max = np.amax(np.absolute(e))
        print(e_max)
        assert e_max < 0.1


if __name__ == "__main__":
    # test_binary_xor()
    #test_categorical_xor()
    test_training_conv1d()
    #test_training_conv1d_with_known_weights()
    print("SUCCESS.")
