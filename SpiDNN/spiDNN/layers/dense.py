import numpy as np

from spinn_utilities.overrides import overrides

from .abstract_layer_base import AbstractLayerBase
from .layer_interface import LayerInterface
from .weights_interface import WeightsInterface

import spiDNN.globals as globals
from spiDNN.machine_vertices import Perceptron


class Dense(AbstractLayerBase, WeightsInterface):
    def __init__(self, n_neurons, activation, bias=True):
        super(Dense, self).__init__("unnamed", n_neurons, [])

        if activation in globals.activations:
            self.activation = activation
        else:
            raise KeyError(
                "Unexpected activation function: {}".format(activation))

        self.bias = bias

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        weights = kwargs["weights"]
        biases = kwargs["biases"]
        trainable_params = kwargs["trainable_params"]

        assert weights.shape[1] == self.n_neurons
        assert biases.shape[0] == self.n_neurons

        for i, weight_vector in enumerate(
                np.concatenate((weights, biases.reshape(1, -1))).T):

            neuron = Perceptron(self, i, weight_vector, trainable_params)

            self.neurons.append(neuron)

        super(Dense, self).init_neurons()

        if self.activation == "softmax":
            self.connect_incoming(self, globals.softmax_partition)

    @overrides(WeightsInterface.generate_weights)
    def generate_weights(self, source_layer):
        source_neurons = source_layer.n_neurons

        weights = np.array(
            np.random.rand(source_neurons, self.n_neurons), dtype=np.float32)
        biases = np.array(
            np.random.rand(self.n_neurons), dtype=np.float32)

        if not self.bias:
            biases[:] = .0

        return weights, biases

    @overrides(WeightsInterface.extract_weights)
    def extract_weights(self):
        weights = np.empty(
            (self.neurons[0].weights.shape[0] - 1, self.n_neurons),
            dtype=np.float32)
        biases = np.empty((self.n_neurons,), dtype=np.float32)

        for i, neuron in enumerate(self.neurons):
            neuron_weights = neuron.extract_weights()

            weights[:, i] = neuron_weights[:-1]
            biases[i] = neuron_weights[-1]

        return weights, biases
