import numpy as np

import math

from spinn_utilities.overrides import overrides

from .abstract_layer_base import AbstractLayerBase
from .layer_interface import LayerInterface
from .weights_interface import WeightsInterface

import spiDNN.globals as globals
import spiDNN.gfe as gfe
from spiDNN.machine_vertices import Conv1DNeuron, Conv1DMeta


class Conv1D(AbstractLayerBase, WeightsInterface):
    def __init__(
            self, n_filters, kernel_shape, activation="identity",
            bias=True, padding="valid", stride=1):

        assert len(kernel_shape) == 1
        # 0 and negative kernel shape is illegal
        assert kernel_shape[0] > 0

        if stride >= kernel_shape[0]:
            raise Exception("""Currently the stride can't equal or
                exceed the kernel_shape""")

        self.kernel_shape = kernel_shape

        self.n_channels = None # set during generate_weights
        self.meta_vertex = None

        self.bias = bias
        self.stride = stride

        if padding in globals.paddings:
            self.padding = padding
        else:
            raise KeyError(
                "Unexpected padding: {}".format(padding))

        if activation in globals.activations:
            self.activation = activation
        else:
            raise KeyError(
                "Unexpected activation function: {}".format(activation))

        super(Conv1D, self).__init__(
            "unnamed", None, [], n_filters=n_filters)

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        weights = kwargs["weights"]
        biases = kwargs["biases"]
        trainable_params = kwargs["trainable_params"]

        weight_vector = np.empty((
            self.kernel_shape[0] * self.n_channels + 1, self.n_filters))

        for i in range(0, self.n_filters):
            filter = weights[:, :, i]
            filter = np.append(filter.flatten(), biases[i])
            weight_vector[:, i] = filter

        weight_vector = weight_vector.flatten(order="F")

        """
        self.meta_vertex = Conv1DMeta(
            self, weight_vector, trainable_params)
        gfe.add_machine_vertex_instance(self.meta_vertex)
        """

        for i in range(0, self.n_neurons):
            neuron = Conv1DNeuron(
                self, i, weight_vector, trainable_params)
            self.neurons.append(neuron)

        super(Conv1D, self).init_neurons()

        """
        # meta conn
        for neuron in self.neurons:
            gfe.add_machine_edge_instance(
                self.meta_vertex, neuron, globals.meta_partition)
            gfe.add_machine_edge_instance(
                neuron, self.meta_vertex, globals.meta_partition)

        if self.activation == "softmax":
            super(Conv1D, self).connect_incoming(
                self, globals.softmax_partition)
        """

    @overrides(LayerInterface.connect_incoming)
    def connect_incoming(self, source_layer, partition):
        if self.padding == "valid":
            growing_down = 0
        elif self.padding == "same":
            size = (self.n_neurons - 1) * self.stride + 1

            padding = self.kernel_shape[0] - 1
            padding -= source_layer.n_neurons - size

            growing_down = int(padding / 2)
        else:
            raise KeyError(
                "Unexpected padding: {}".format(padding))

        i = -growing_down
        for neuron in self.neurons:
            if i < 0:
                neuron.lower_padding = abs(i)

            if i + self.kernel_shape[0] >= source_layer.n_neurons:
                neuron.upper_padding = \
                    i + self.kernel_shape[0] - source_layer.n_neurons

            for j in range(
                    i + neuron.lower_padding,
                    i + self.kernel_shape[0] - neuron.upper_padding):
                gfe.add_machine_edge_instance(
                    source_layer.neurons[j], neuron, partition)
            i += self.stride

    @overrides(WeightsInterface.generate_weights)
    def generate_weights(self, source_layer):
        self._set_n_neurons(source_layer)

        self.n_channels = source_layer.n_filters

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

    def _set_n_neurons(self, source_layer):
        if self.padding == "valid":
            self.n_neurons = int(
                (source_layer.n_neurons - self.kernel_shape[0])
                / self.stride + 1)

            size = (self.n_neurons - 1) * self.stride + 1
            size += 2 * int(self.kernel_shape[0] / 2)

            print(self, self.n_neurons, source_layer.n_neurons, size)

            if size < source_layer.n_neurons:
                raise Exception("""{}: The chosen stride of {} with
                    valid padding currently is not allowed, because
                    not all neurons from the previous layer will be
                    connected.""".format(self, self.stride))
        else:
            self.n_neurons = \
                int(math.ceil(source_layer.n_neurons / self.stride))
