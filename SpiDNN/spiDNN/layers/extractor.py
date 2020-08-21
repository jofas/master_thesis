from spinn_front_end_common.utility_models import \
    LivePacketGatherMachineVertex
from spinn_front_end_common.utilities.utility_objs import \
    LivePacketGatherParameters
from pacman.model.constraints.placer_constraints import \
    ChipAndCoreConstraint
from spinnman.messages.eieio import EIEIOType

import spiDNN.globals as globals

from .abstract_layer_base import AbstractLayerBase


class Extractor(AbstractLayerBase):
    """
    This class is an abstraction over the LPG. This abstraction is
    needed so we don't violate our layered design pattern.
    """

    def __init__(self, label):
        args = LivePacketGatherParameters(
            port=globals.ack_port,
            hostname=globals.host,
            strip_sdp=True,
            message_type=EIEIOType.KEY_PAYLOAD_32_BIT,
            use_payload_prefix=False,
            payload_as_time_stamps=False)

        machine_vertex = LivePacketGatherMachineVertex(
            args, label=label, constraints=[ChipAndCoreConstraint(x=0, y=0)])

        super(Extractor, self).__init__(label, 1, [machine_vertex])
