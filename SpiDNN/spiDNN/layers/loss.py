import spinnaker_graph_front_end as front_end

from pacman.model.graphs.machine import MachineEdge


from spiDNN.machine_vertices import LossMachineVertex

import spiDNN.globals as globals


class Loss:
    def __init__(self, loss_fn, label):
        self.loss_fn = loss_fn
        self.label = label

        # TODO: here add machine_vertex instance
        self.machine_vertex = LossMachineVertex(self)

    def connect(self, source_layer):
        for source_neuron in source_layer.neurons:
            front_end.add_machine_edge_instance(MachineEdge(
                source_neuron, self.machine_vertex,
                label="{}_to_{}".format(
                    source_neuron.label, self.machine_vertex.label)
                ), globals.partition_name)
