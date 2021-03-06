from spinn_utilities.overrides import overrides


from .abstract_layer_base import AbstractLayerBase

from .layer_interface import LayerInterface

import spiDNN.gfe as gfe
from spiDNN.machine_vertices import Injector


class Input(AbstractLayerBase):
    """
    Layer for streaming the observations onto the board. Neurons
    are Injectors (wrapper around the RIPTMCS utility vertex from the
    SpiNNaker toolchain).
    """

    def __init__(self, *input_shape, label="unnamed"):
        n_neurons = input_shape[0]

        if len(input_shape) == 1:
            n_filters = 1  # actually the amount of channels
        else:
            n_filters = input_shape[-1]  # channel_last

        super(Input, self).__init__(
            label, n_neurons, [], n_filters=n_filters)

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        neurons_next_layer = kwargs["neurons_next_layer"]

        for i in range(0, self.n_neurons):
            neuron = Injector(self, i, neurons_next_layer)
            self.neurons.append(neuron)
            gfe.add_machine_vertex_instance(neuron)
