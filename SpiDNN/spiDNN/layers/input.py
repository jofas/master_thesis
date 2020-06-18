import spinnaker_graph_front_end as front_end

from spinn_utilities.overrides import overrides


from .abstract_layer_base import AbstractLayerBase

from .layer_interface import LayerInterface


from spiDNN.machine_vertices import Injector


class Input(AbstractLayerBase):
    def __init__(self, n_neurons):
        super(Input, self).__init__("unnamed", n_neurons, [])

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        neurons_next_layer = kwargs["neurons_next_layer"]
        partition_manager = kwargs["partition_manager"]

        for i in range(0, self.n_neurons):
            neuron = Injector(
                n_keys=neurons_next_layer,
                label="{}_{}".format(self.label, i),
                partition_manager=partition_manager)
            self.neurons.append(neuron)
            front_end.add_machine_vertex_instance(neuron)
