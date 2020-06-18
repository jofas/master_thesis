import spinnaker_graph_front_end as front_end

from spinn_utilities.overrides import overrides


from .layer_interface import LayerInterface


import spiDNN.util as util


class AbstractLayerBase(LayerInterface):
    def __init__(self, label, n_neurons, neurons):
        self._label = label
        self._n_neurons = n_neurons
        self._neurons = neurons

    @overrides(LayerInterface.connect_incoming)
    def connect_incoming(self, source_layer, partition):
        for source_neuron in source_layer.neurons:
            for neuron in self.neurons:
                util.add_machine_edge_instance(
                    source_neuron, neuron, partition)

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        for neuron in self.neurons:
            front_end.add_machine_vertex_instance(neuron)

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
    def label(self, new_label):
        self._label = new_label

    @property
    @overrides(LayerInterface.labels)
    def labels(self):
        return [neuron.label for neuron in self._neurons]
