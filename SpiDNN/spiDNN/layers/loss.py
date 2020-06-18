import spinnaker_graph_front_end as front_end


from spiDNN.machine_vertices import LossMachineVertex

import spiDNN.globals as globals

import spiDNN.util as util


class Loss:
    """
    Wrapper class for a LossMachineVertex.
    """

    def __init__(self, label, loss_fn, K):
        self.machine_vertex = LossMachineVertex(label, loss_fn, K)

    def init_neurons(self):
        front_end.add_machine_vertex_instance(self.machine_vertex)

    def connect_incoming(
            self, source_layer, partition=globals.forward_partition):
        """
        Connect Loss layer instance with the output layer of the
        neural network.
        """
        for source_neuron in source_layer.neurons:
            util.add_machine_edge_instance(
                source_neuron, self.machine_vertex, partition)
