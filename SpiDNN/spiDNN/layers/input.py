from spinn_utilities.overrides import overrides


from .abstract_layer_base import AbstractLayerBase

from .layer_interface import LayerInterface

import spiDNN.gfe as gfe
from spiDNN.machine_vertices import Injector


class Input(AbstractLayerBase):
    def __init__(self, *input_shape, label="unnamed"):
        n_neurons = input_shape[0]
        self.n_filters = 1  # actually the amount of channels
        super(Input, self).__init__(label, n_neurons, [])

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        neurons_next_layer = kwargs["neurons_next_layer"]

        for i in range(0, self.n_neurons):
            neuron = Injector(self, i, neurons_next_layer)
            self.neurons.append(neuron)
            gfe.add_machine_vertex_instance(neuron)
