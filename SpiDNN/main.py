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

    kmodel.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    kmodel.fit(X, y, epochs=1000, batch_size=4)

    model = Model().add(Input(2)) \
                   .add(Dense(2, activation="relu")) \
                   .add(Dense(1, activation="sigmoid"))

    hidden_weights = kmodel.weights[0].read_value().numpy().T
    hidden_bias = kmodel.weights[1].read_value().numpy().reshape(-1, 1)

    output_weights = kmodel.weights[2].read_value().numpy().T
    output_bias = kmodel.weights[3].read_value().numpy().reshape(-1, 1)

    model.weights[0] = np.concatenate(
        (hidden_weights, hidden_bias), axis=1
    )

    model.weights[1] = np.concatenate(
        (output_weights, output_bias), axis=1
    )

    # LOLZ keras neurons are columns
    """
    # weights of hidden layer
    model.weights[0] = np.array([
        [-1.68221831,  0.75817555, -4.67257014e-05],
        [ 1.68205309, -0.75822848, -4.66354031e-05]
    ], dtype=np.float32)

    # weigths of output layer
    model.weights[1] = np.array([
        [1.10278344, 1.97492659, -0.48494098]
    ], dtype=np.float32)
    """

    prediction = model.predict(X)

    print(prediction)

    p_ = kmodel.predict(X, batch_size=4)
    print(p_)
    # print(np.sum(p_, axis=1))

    # print(hidden_weights)
    # print(hidden_bias)


if __name__ == "__main__":
    main()
