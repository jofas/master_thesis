import numpy as np

import math

from spinn_utilities.overrides import overrides

from .abstract_layer_base import AbstractLayerBase
from .layer_interface import LayerInterface
from .weights_interface import WeightsInterface

import spiDNN.globals as globals
import spiDNN.gfe as gfe
from spiDNN.machine_vertices import Conv1DNeuron
import spinnaker_graph_front_end as front_end


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

        for i in range(0, self.n_neurons):
            neuron = Conv1DNeuron(
                self, i, weight_vector, trainable_params)
            self.neurons.append(neuron)

        super(Conv1D, self).init_neurons()

        if trainable_params is not None:
            super(Conv1D, self).connect_incoming(
                self, globals.kernel_update_partition)

    @overrides(LayerInterface.connect_incoming)
    def connect_incoming(self, source_layer, partition):
        growing_down = self._get_growing_down(source_layer)

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

    @overrides(LayerInterface.connect_outgoing)
    def connect_outgoing(self, dest_layer, partition):
        growing_down = self._get_growing_down(dest_layer)

        i = -growing_down
        for neuron in self.neurons:
            for j in range(
                    i + neuron.lower_padding,
                    i + self.kernel_shape[0] - neuron.upper_padding):
                gfe.add_machine_edge_instance(
                    neuron, dest_layer.neurons[j], partition)
            i += self.stride

        """
        for dest_neuron in dest_layer.neurons:
            for neuron in self.neurons:
                in_lower_bound = \
                    neuron.id - neuron.lower_padding <= dest_neuron.id

                in_upper_bound = dest_neuron.id < \
                    neuron.id + self.kernel_shape[0] - neuron.upper_padding

                if in_upper_bound and in_lower_bound:
                    gfe.add_machine_edge_instance(
                        neuron, dest_neuron, partition)
        """

    def _get_growing_down(self, other_layer):
        if self.padding == "valid":
            return 0
        elif self.padding == "same":
            size = (self.n_neurons - 1) * self.stride + 1

            padding = self.kernel_shape[0] - 1
            padding -= other_layer.n_neurons - size

            return int(padding / 2)
        else:
            raise KeyError(
                "Unexpected padding: {}".format(padding))

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
        weights = np.empty(
            (*self.kernel_shape, self.n_channels, self.n_filters),
            dtype=np.float32)
        biases = np.empty((self.n_filters,), dtype=np.float32)

        flattened_weights = self.neurons[0].extract_weights()

        flattened_weights = flattened_weights.reshape(
            self.kernel_shape[0] * self.n_channels + 1, self.n_filters,
            order="F")

        for i in range(0, self.n_filters):
            weights[:, :, i] = flattened_weights[:-1, i].reshape(
                self.kernel_shape[0], self.n_channels)
            biases[i] = flattened_weights[-1, i]

        return weights, biases

    def _set_n_neurons(self, source_layer):
        if self.padding == "valid":
            self.n_neurons = int(
                (source_layer.n_neurons - self.kernel_shape[0])
                / self.stride + 1)

            size = (self.n_neurons - 1) * self.stride + 1
            size += 2 * int(self.kernel_shape[0] / 2)

            if size < source_layer.n_neurons:
                raise Exception("""{}: The chosen stride of {} with
                    valid padding currently is not allowed, because
                    not all neurons from the previous layer will be
                    connected.""".format(self, self.stride))
        else:
            self.n_neurons = \
                int(math.ceil(source_layer.n_neurons / self.stride))
