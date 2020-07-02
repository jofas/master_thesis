import numpy as np

from spinn_utilities.overrides import overrides

from .abstract_layer_base import AbstractLayerBase
from .layer_interface import LayerInterface
from .weights_interface import WeightsInterface

import spiDNN.globals as globals
from spiDNN.machine_vertices import Perceptron


class Conv1D(AbstractLayerBase, WeightsInterface):
    def __init__(self, kernel_shape, activation, bias=True):
        assert len(kernel_shape) == 1

        self.kernel_shape = kernel_shape

        # TODO: support n filters
        self.n_filters = 1
        self.n_channels = None
        # TODO: support "same" as well
        self.padding = "valid"

        super(Conv1D, self).__init__("unnamed", None, [])

        if activation in globals.activations:
            self.activation = activation
        else:
            raise KeyError(
                "Unexpected activation function: {}".format(activation))

        self.bias = bias

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        pass
        """
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
        """

    @overrides(WeightsInterface.generate_weights)
    def generate_weights(self, source_layer):
        # TODO: set self.n_neurons depending on source_layer and
        #       self.padding

        # TODO: channel interface in input
        #self.n_channels = source_layer.n_channels
        self.n_channels = 1

        weights = np.array(np.random.rand(
            *self.kernel_shape, self.n_channels, self.n_filters),
            dtype=np.float32)

        biases = np.array(np.random.rand(
            self.n_filters), dtype=np.float32)

        if not self.bias:
            biases[:] = .0

        return weights, biases

    @overrides(WeightsInterface.extract_weights)
    def extract_weights(self):
        pass
        """
        weights = np.empty(
            (self.neurons[0].weights.shape[0] - 1, self.n_neurons),
            dtype=np.float32)
        biases = np.empty((self.n_neurons,), dtype=np.float32)

        for i, neuron in enumerate(self.neurons):
            neuron_weights = neuron.extract_weights()

            weights[:, i] = neuron_weights[:-1]
            biases[i] = neuron_weights[-1]

        return weights, biases
        """
