from pacman.model.graphs.machine import MachineVertex
from spinn_utilities.overrides import overrides
from spinn_front_end_common.abstract_models import \
    AbstractProvidesOutgoingPartitionConstraints, \
    AbstractProvidesNKeysForPartition

import spiDNN.gfe as gfe


class AbstractPartitionManagedMachineVertex(
        MachineVertex,
        AbstractProvidesOutgoingPartitionConstraints,
        AbstractProvidesNKeysForPartition):

    @overrides(AbstractProvidesOutgoingPartitionConstraints
               .get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return gfe.partition_manager.generate_constraints(
            self, partition.identifier)

    @overrides(AbstractProvidesNKeysForPartition
               .get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition):
        return 1
