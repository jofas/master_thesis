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


class Extractor:
    def __init__(self, label):
        self.label = label

        args = LivePacketGatherParameters(
            port=globals.ack_port,
            hostname=globals.host,
            strip_sdp=True,
            message_type=EIEIOType.KEY_PAYLOAD_32_BIT,
            use_payload_prefix=False,
            payload_as_time_stamps=False)

        self.machine_vertex = LivePacketGatherMachineVertex(
            args, self.label, constraints=[ChipAndCoreConstraint(x=0, y=0)])

        front_end.add_machine_vertex_instance(self.machine_vertex)

    def connect(self, source_layer):
        for source_neuron in source_layer.neurons:
            front_end.add_machine_edge_instance(MachineEdge(
                source_neuron, self.machine_vertex,
                label="{}_to_{}".format(
                    source_neuron.label, self.machine_vertex.label)
                ), globals.forward_partition)
