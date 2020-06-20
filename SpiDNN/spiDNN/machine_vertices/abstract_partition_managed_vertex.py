from spinn_utilities.overrides import overrides
from spinn_front_end_common.abstract_models import \
    AbstractProvidesOutgoingPartitionConstraints

import spiDNN.gfe as gfe

# TODO: inheritance for each machine_vertex through this one
class AbstractPartitionManagedVertex(
        AbstractProvidesOutgoingPartitionConstraints):

    @overrides(AbstractProvidesOutgoingPartitionConstraints
               .get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [gfe.partition_manager.generate_constraint(
            self, partition.identifier)]
