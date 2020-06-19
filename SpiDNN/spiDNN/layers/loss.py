from spinn_utilities.overrides import overrides


from spiDNN.machine_vertices import LossMachineVertex

import spiDNN.globals as globals


from .abstract_layer_base import AbstractLayerBase

from .layer_interface import LayerInterface


class Loss(AbstractLayerBase):
    def __init__(self, label, loss_fn, K):
        super(Loss, self).__init__(label, 1, [])

        self.loss_fn = loss_fn
        self.K = K

    @overrides(LayerInterface.init_neurons)
    def init_neurons(self, **kwargs):
        partition_manager = kwargs["partition_manager"]
        machine_vertex = LossMachineVertex(self, partition_manager)
        self.neurons.append(machine_vertex)
        super(Loss, self).init_neurons()
