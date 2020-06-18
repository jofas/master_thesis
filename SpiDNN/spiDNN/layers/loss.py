import spinnaker_graph_front_end as front_end

from pacman.model.graphs.machine import MachineEdge


from spiDNN.machine_vertices import LossMachineVertex

import spiDNN.globals as globals


class Loss:
    """
    Wrapper class for a LossMachineVertex.
    """

    def __init__(self, label, loss_fn, K):
        self.machine_vertex = LossMachineVertex(label, loss_fn, K)

    def connect(self, source_layer, partition=globals.forward_partition):
        """
        Connect Loss layer instance with the output layer of the
        neural network.
        """
        for source_neuron in source_layer.neurons:
            front_end.add_machine_edge_instance(MachineEdge(
                source_neuron, self.machine_vertex,
                label="{}_to_{}".format(
                    source_neuron.label, self.machine_vertex.label)
                ), partition)
