from spiDNN.layers import Conv1D, Input

from keras.layers import Conv1D as KConv1D

import tensorflow as tf


def test_weight_generation():
    n_batches = 5
    batch_size = 8
    n_channels = 1
    kernel_shape = (2,)
    input_shape = (n_batches, batch_size, n_channels)

    X = tf.random.normal(input_shape)

    kconv = KConv1D(1, kernel_shape, input_shape=input_shape[1:])
    conv = Conv1D(kernel_shape, "identity")

    # generate weights
    kconv(X)
    keras_weights_and_biases = kconv.get_weights()
    weights, biases = conv.generate_weights(Input(5))

    assert keras_weights_and_biases[0].shape == weights.shape
    assert keras_weights_and_biases[1].shape == biases.shape


if __name__ == "__main__":
    test_weight_generation()
    print("SUCCESS.")
