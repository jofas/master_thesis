from spinn_front_end_common.utility_models import \
    ReverseIPTagMulticastSourceMachineVertex


from .abstract_partition_managed_vertex import AbstractPartitionManagedVertex


class Injector(
        AbstractPartitionManagedVertex,
        ReverseIPTagMulticastSourceMachineVertex):

    def __init__(self, n_keys, label, partition_manager):
        AbstractPartitionManagedVertex.__init__(
            self, partition_manager=partition_manager)

        ReverseIPTagMulticastSourceMachineVertex.__init__(
            self, n_keys=n_keys, label=label, enable_injection=True)
