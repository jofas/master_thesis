import numpy as np


import spinnaker_graph_front_end as front_end

from pacman.model.graphs.machine import MachineEdge


import spiDNN.globals as globals

from .neurons import Perceptron


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

        for i, weight_vector in enumerate(
                np.concatenate((weights, biases.reshape(1, -1))).T):

            neuron = Perceptron(self, i, weight_vector)
            self.neurons.append(neuron)
            front_end.add_machine_vertex_instance(neuron)

    def connect(self, source_layer):
        for source_neuron in source_layer.neurons:
            for neuron in self.neurons:
                # neuron needs to know what previous layer activation
                # is, because it needs to do some work if previous
                # layer has either ReLU or softmax as activation.
                # Both are layer-based rather than on a neuron level.
                neuron.set_pre_layer_activation(source_layer)

                front_end.add_machine_edge_instance(MachineEdge(
                    source_neuron, neuron, label="{}_to_{}".format(
                        source_neuron.label, neuron.label
                    )
                ), globals.partition_name)

    def generate_weights(self, source_layer):
        # This is just weights representation. The weights are re-
        # injected to the neuron right before starting the simulation.
        # Called in Model.add().
        #
        source_atoms = source_layer.atoms

        weights = np.array(
            np.random.rand(source_atoms, self.atoms), dtype=np.float32
        )
        biases = np.array(
            np.random.rand(self.atoms), dtype=np.float32
        )

        if not self.bias:
            biases[:] = .0

        return [weights, biases]

    @property
    def labels(self):
        return [neuron.label for neuron in self.neurons]
