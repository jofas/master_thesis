from spinn_utilities.overrides import overrides

from spinn_front_end_common.abstract_models import \
    AbstractProvidesOutgoingPartitionConstraints


class AbstractPartitionManagedVertex(
        AbstractProvidesOutgoingPartitionConstraints):
    def __init__(self, partition_manager):
        self.partition_manager = partition_manager

    @overrides(AbstractProvidesOutgoingPartitionConstraints
               .get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        return [self.partition_manager.generate_constraint(
            partition.identifier)]
