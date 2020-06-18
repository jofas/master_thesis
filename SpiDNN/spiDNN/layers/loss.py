from spiDNN.machine_vertices import LossMachineVertex

import spiDNN.globals as globals


from .abstract_layer_base import AbstractLayerBase


class Loss(AbstractLayerBase):
    def __init__(self, label, loss_fn, K):
        machine_vertex = LossMachineVertex(label, loss_fn, K)
        super(Loss, self).__init__(label, 1, [machine_vertex])
