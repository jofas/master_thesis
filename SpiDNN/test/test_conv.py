import spinnaker_graph_front_end as front_end

from spiDNN import Model
from spiDNN.layers import Conv1D, Input, Dense
import spiDNN.gfe as gfe

import tensorflow as tf

import numpy as np


# column -> one filter
weights = np.array([[[0.0, 1.0, 2.0],
                     [0.1, 1.1, 2.1],
                     [0.2, 1.2, 2.2]],
                    [[0.3, 1.3, 2.3],
                     [0.4, 1.4, 2.4],
                     [0.5, 1.5, 2.5]],
                    [[0.6, 1.6, 2.6],
                     [0.7, 1.7, 2.7],
                     [0.8, 1.8, 2.8]]], dtype=np.float32)

biases = np.array([0.9, 1.9, 2.9], dtype=np.float32)

kernel_size = 3
n_channels = 3
n_filters = 3


def test_conv_flatten():
    assert (weights[:, :, 0] == np.array([[0.0, 0.1, 0.2],
                                          [0.3, 0.4, 0.5],
                                          [0.6, 0.7, 0.8]],
                                         dtype=np.float32)).all()

    conv1d = Conv1D(1, (kernel_size,))
    conv1d.n_filters = n_filters

    input = Input(2)
    input.n_filters = n_channels

    weights_, biases_ = conv1d.generate_weights(input)

    assert weights.shape == weights_.shape
    assert biases.shape == biases_.shape

    flattened_weights = np.empty((
        kernel_size * n_channels + 1, n_filters))

    for i in range(0, n_filters):
        filter = weights[:, :, i]
        filter = np.append(filter.flatten(), biases[i])
        flattened_weights[:, i] = filter

    flattened_weights = flattened_weights.flatten(order="F")

    flattened_weights = flattened_weights.reshape(
        kernel_size * n_channels + 1, n_filters, order="F")

    extracted_weights = np.empty((kernel_size, n_channels, n_filters))
    extracted_biases = np.empty((n_filters,))

    for i in range(0, n_filters):
        extracted_weights[:, :, i] = flattened_weights[:-1, i].reshape(
            kernel_size, n_channels)
        extracted_biases[i] = flattened_weights[-1, i]

    assert (extracted_biases == biases).all()
    assert (extracted_weights == weights).all()


def test_convolution():
    input_shape = (1, kernel_size, n_channels)

    # rows are channels, columns are the input vectors
    input = np.array([[[1., 0., 1.],
                       [0., 1., 0.],
                       [0., 0., 1.]]], dtype=np.float32)

    assert input.shape == input_shape

    layer = tf.keras.layers.Conv1D(
        n_filters, kernel_size, input_shape=input_shape[1:])
    layer(input)
    layer.set_weights([weights, biases])
    y = layer(input).numpy()

    correct_result = np.array([1.2 + 0.2 + biases[0],
                               4.2 + 1.2 + biases[1],
                               7.2 + 2.2 + biases[2]], dtype=np.float32)

    assert np.amax(np.absolute(y[0,0,:] - correct_result)) < 1e-6


def test_connection():
    input_layer = Input(5)
    input_layer.label = "Input"

    conv1d_layer = Conv1D(1, (3,))
    conv1d_layer.label = "Conv1D"

    weights, biases = conv1d_layer.generate_weights(input_layer)

    gfe.setup(input_layer.n_neurons + conv1d_layer.n_neurons)

    input_layer.init_neurons(neurons_next_layer=conv1d_layer.n_neurons)

    conv1d_layer.init_neurons(
        weights=weights, biases=biases, trainable_params=None)

    conv1d_layer.connect_incoming(input_layer, "some_partition")

    try:
        gfe.run(1)
    except:
        pass

    assert conv1d_layer.n_neurons == 3

    for neuron in conv1d_layer.neurons:
        edges = list(filter(
            lambda x: x.post_vertex == neuron
            and x.label.startswith("some_partition"),
            front_end.machine_graph().edges))

        assert len(edges) == conv1d_layer.kernel_shape[0]

        for j in range(0, conv1d_layer.kernel_shape[0]):
            edge = list(filter(
                lambda x: x.pre_vertex.id == neuron.id + j, edges))
            assert len(edge) == 1

    try:
        gfe.stop()
    except:
        pass


def test_same_padding():
    model = Model().add(Input(5)) \
                   .add(Conv1D(1, (8,), padding="same")) \
                   .add(Dense(1, activation="identity"))

    X = np.random.rand(1, 5, 1)

    model.predict(X)

    lower, upper = \
        model._layers[1].neurons[0]._generate_lower_and_upper_padding()
    assert lower == 3 and upper == 0

    lower, upper = \
        model._layers[1].neurons[1]._generate_lower_and_upper_padding()
    assert lower == 2 and upper == 1

    lower, upper = \
        model._layers[1].neurons[2]._generate_lower_and_upper_padding()
    assert lower == 1 and upper == 2

    lower, upper = \
        model._layers[1].neurons[3]._generate_lower_and_upper_padding()
    assert lower == 0 and upper == 3

    lower, upper = \
        model._layers[1].neurons[4]._generate_lower_and_upper_padding()
    assert lower == 0 and upper == 4


if __name__ == "__main__":
    #test_conv_flatten()
    #test_convolution()
    #test_connection()
    test_same_padding()
    print("SUCCESS.")
