import spinnaker_graph_front_end as front_end

from spinn_front_end_common.utility_models import \
    ReverseIPTagMulticastSourceMachineVertex

from spinn_utilities.overrides import overrides


from .abstract_layer_base import AbstractLayerBase

from .layer_interface import LayerInterface


class Input(AbstractLayerBase):
    def __init__(self, n_neurons):
        super(Input, self).__init__("unnamed", n_neurons, [])

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        neurons_next_layer = kwargs["neurons_next_layer"]

        for i in range(0, self._n_neurons):
            neuron = ReverseIPTagMulticastSourceMachineVertex(
                n_keys=neurons_next_layer,
                label="{}_{}".format(self.label, i),
                enable_injection=True,
            )
            self._neurons.append(neuron)
            front_end.add_machine_vertex_instance(neuron)
