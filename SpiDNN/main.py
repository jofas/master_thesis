from spiDNN import Model
from spiDNN.layers import Input, Dense

import numpy as np


def main():
    model = Model().add(Input(2)) \
                   .add(Dense(2, activation="sigmoid")) \
                   .add(Dense(2, activation="sigmoid"))

    print(model.weights)

    test = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

    result = np.array([0, 1, 1, 0])

    prediction = model.predict(test)

    print(prediction)


if __name__ == "__main__":
    main()
