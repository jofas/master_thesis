import numpy as np


import spinnaker_graph_front_end as front_end

from pacman.model.graphs.machine import MachineEdge


import spiDNN.globals as globals

from spiDNN.machine_vertices import Perceptron, SoftmaxPerceptron


class Dense:
    def __init__(self, atoms, activation, bias=True):
        self.atoms = atoms
        self.name = "uninitialized"
        self.neurons = []

        if activation in globals.activations:
            self.activation = activation
        else:
            raise KeyError(
                "Unexpected activation function: {}".format(activation)
            )

        self.bias = bias

    def init_neurons(self, weights, biases):
        assert weights.shape[1] == self.atoms
        assert biases.shape[0] == self.atoms

        if self.activation == "softmax":
            self._init_softmax_perceptron(weights, biases)
        else:
            self._init_perceptron(weights, biases)

    def _init_softmax_perceptron(self, weights, biases):
        for i, weight_vector in enumerate(
                np.concatenate((weights, biases.reshape(1, -1))).T):

            neuron = SoftmaxPerceptron(self, i, weight_vector)
            self.neurons.append(neuron)
            front_end.add_machine_vertex_instance(neuron)

        self._connect_softmax_perceptrons()

    def _init_perceptron(self, weights, biases):
        for i, weight_vector in enumerate(
                np.concatenate((weights, biases.reshape(1, -1))).T):

            neuron = Perceptron(self, i, weight_vector)
            self.neurons.append(neuron)
            front_end.add_machine_vertex_instance(neuron)

    def _connect_softmax_perceptrons(self):
        """
        Connect each neuron in this layer to all other neurons except
        itself in a partition unique to this layer.
        """
        for source_neuron in self.neurons:
            for neuron in self.neurons:
                if source_neuron == neuron:
                    continue

                front_end.add_machine_edge_instance(MachineEdge(
                    source_neuron, neuron,
                    label="{}_softmax_{}_to_{}".format(
                        self.name, source_neuron.label, neuron.label)
                ), globals.softmax_partition)

    def connect(self, source_layer):
        for source_neuron in source_layer.neurons:
            for neuron in self.neurons:
                front_end.add_machine_edge_instance(MachineEdge(
                    source_neuron, neuron, label="{}_to_{}".format(
                        source_neuron.label, neuron.label)
                ), globals.forward_partition)

    def generate_weights(self, source_layer):
        # This is just weights representation. The weights are re-
        # injected to the neuron right before starting the simulation.
        # Called in Model.add().
        #
        source_atoms = source_layer.atoms

        weights = np.array(
            np.random.rand(source_atoms, self.atoms), dtype=np.float32)
        biases = np.array(
            np.random.rand(self.atoms), dtype=np.float32)

        if not self.bias:
            biases[:] = .0

        return weights, biases

    def extract_weights(self):
        weights = np.empty(
            (self.neurons[0].weights.shape[0] - 1, self.atoms),
            dtype=np.float32)
        biases = np.empty((self.atoms,), dtype=np.float32)

        for i, neuron in enumerate(self.neurons):
            neuron_weights = neuron.extract_weights(
                front_end.transceiver(),
                front_end.placements().get_placement_of_vertex(neuron),
            )

            weights[:, i] = neuron_weights[:-1]
            biases[i] = neuron_weights[-1]

        return weights, biases

    @property
    def labels(self):
        return [neuron.label for neuron in self.neurons]
