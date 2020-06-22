from spinn_front_end_common.utility_models import \
    ReverseIPTagMulticastSourceMachineVertex

from .abstract_partition_managed_machine_vertex import \
    AbstractPartitionManagedMachineVertex


class Injector(
        AbstractPartitionManagedMachineVertex,
        ReverseIPTagMulticastSourceMachineVertex):

    def __init__(self, n_keys, label):
        ReverseIPTagMulticastSourceMachineVertex.__init__(
            self, n_keys=n_keys, label=label, enable_injection=True)
