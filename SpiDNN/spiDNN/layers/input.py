from spinn_utilities.overrides import overrides


from .abstract_layer_base import AbstractLayerBase

from .layer_interface import LayerInterface

import spiDNN.gfe as gfe
from spiDNN.machine_vertices import Injector


class Input(AbstractLayerBase):
    def __init__(self, n_neurons, label="unnamed"):
        super(Input, self).__init__(label, n_neurons, [])

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        neurons_next_layer = kwargs["neurons_next_layer"]

        for i in range(0, self.n_neurons):
            neuron = Injector(
                n_keys=neurons_next_layer,
                label="{}_{}".format(self.label, i))
            self.neurons.append(neuron)
            gfe.add_machine_vertex_instance(neuron)
