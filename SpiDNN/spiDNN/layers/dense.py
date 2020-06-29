import numpy as np

import spinnaker_graph_front_end as front_end

from spinn_utilities.overrides import overrides


from .abstract_layer_base import AbstractLayerBase
from .layer_interface import LayerInterface

import spiDNN.globals as globals
import spiDNN.util as util
from spiDNN.machine_vertices import Perceptron, SoftmaxPerceptron


class Dense(AbstractLayerBase):
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
        trainable = kwargs["trainable"]
        batch_size = kwargs["batch_size"]
        learning_rate = kwargs["learning_rate"]

        assert weights.shape[1] == self.n_neurons
        assert biases.shape[0] == self.n_neurons

        for i, weight_vector in enumerate(
                np.concatenate((weights, biases.reshape(1, -1))).T):

            if self.activation == "softmax":
                neuron = SoftmaxPerceptron(
                    self, i, weight_vector, trainable, batch_size,
                    learning_rate)
            else:
                neuron = Perceptron(
                    self, i, weight_vector, trainable, batch_size,
                    learning_rate)

            self.neurons.append(neuron)

        super(Dense, self).init_neurons()

        if self.activation == "softmax":
            self.connect_incoming(self, globals.softmax_partition)

    def generate_weights(self, source_layer):
        # This is just weights representation. The weights are re-
        # injected to the neuron right before starting the simulation.
        # Called in Model.add().
        #
        source_neurons = source_layer.n_neurons

        weights = np.array(
            np.random.rand(source_neurons, self.n_neurons), dtype=np.float32)
        biases = np.array(
            np.random.rand(self.n_neurons), dtype=np.float32)

        if not self.bias:
            biases[:] = .0

        return weights, biases

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
