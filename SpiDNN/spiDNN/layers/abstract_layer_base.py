from spinn_utilities.overrides import overrides

from .layer_interface import LayerInterface

import spiDNN.gfe as gfe
import spiDNN.util as util


class AbstractLayerBase(LayerInterface):
    def __init__(self, label, n_neurons, neurons):
        self._label = label
        self._n_neurons = n_neurons
        self._neurons = neurons

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        for neuron in self.neurons:
            gfe.add_machine_vertex_instance(neuron)

    @overrides(LayerInterface.connect_incoming)
    def connect_incoming(self, source_layer, partition):
        for source_neuron in source_layer.neurons:
            for neuron in self.neurons:
                # in case of connecting self with self
                if source_neuron == neuron:
                    continue

                gfe.add_machine_edge_instance(
                    source_neuron, neuron, partition)

    @overrides(LayerInterface.connect_incoming_unique)
    def connect_incoming_unique(self, source_layer, base_name):
        for neuron in self.neurons:
            for source_neuron in source_layer.neurons:
                partition = "{}_{}_to_{}".format(
                    base_name, source_neuron.label, neuron.label)

                gfe.add_machine_edge_instance(
                    source_neuron, neuron, partition)

    @overrides(LayerInterface.reset)
    def reset(self):
        self._neurons = []

    @property
    @overrides(LayerInterface.n_neurons)
    def n_neurons(self):
        return self._n_neurons

    @property
    @overrides(LayerInterface.neurons)
    def neurons(self):
        return self._neurons

    @property
    @overrides(LayerInterface.label)
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    @overrides(LayerInterface.labels)
    def labels(self):
        return [neuron.label for neuron in self._neurons]
