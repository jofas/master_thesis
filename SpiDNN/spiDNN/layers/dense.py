import numpy as np


import spinnaker_graph_front_end as front_end

from pacman.model.graphs.machine import MachineEdge


import spiDNN.globals as globals

from .neurons import Perceptron


class Dense:
    def __init__(self, atoms, activation, bias = True):
        self.atoms = atoms
        self.name = "uninitialized"
        self.neurons = []

        self.activation = activation
        self.bias = bias


    def init_neurons(self, weights):
        assert weights.shape[0] == self.atoms

        for i, weight_vector in enumerate(weights):
            neuron = Perceptron(self, i, weight_vector)
            self.neurons.append(neuron)
            front_end.add_machine_vertex_instance(neuron)


    def connect(self, source_layer):
        for source_neuron in source_layer.neurons:
            for neuron in self.neurons:
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

        weights = np.random.rand(self.atoms, source_atoms + 1)

        if not self.bias:
            weights[:, -1] = 0.

        return weights


    @property
    def labels(self):
        return [neuron.label for neuron in self.neurons]
