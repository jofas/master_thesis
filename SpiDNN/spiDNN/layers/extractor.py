import spinnaker_graph_front_end as front_end

from spinn_front_end_common.utility_models import \
    LivePacketGatherMachineVertex

from spinn_front_end_common.utilities.utility_objs import \
    LivePacketGatherParameters

from pacman.model.constraints.placer_constraints import \
    ChipAndCoreConstraint

from pacman.model.graphs.machine import MachineEdge

from spinnman.messages.eieio import EIEIOType


import spiDNN.globals as globals

import spiDNN.util as util


class Extractor:
    def __init__(self, label):
        args = LivePacketGatherParameters(
            port=globals.ack_port,
            hostname=globals.host,
            strip_sdp=True,
            message_type=EIEIOType.KEY_PAYLOAD_32_BIT,
            use_payload_prefix=False,
            payload_as_time_stamps=False)

        self.machine_vertex = LivePacketGatherMachineVertex(
            args, label, constraints=[ChipAndCoreConstraint(x=0, y=0)])

    def init_neurons(self):
        front_end.add_machine_vertex_instance(self.machine_vertex)

    def connect_incoming(
            self, source_layer, partition=globals.forward_partition):
        for source_neuron in source_layer.neurons:
            util.add_machine_edge_instance(
                source_neuron, self.machine_vertex, partition)

    @property
    def labels(self):
        return [self.machine_vertex.label]
