from spinn_front_end_common.utility_models import \
    ReverseIPTagMulticastSourceMachineVertex


from .abstract_partition_managed_vertex import AbstractPartitionManagedVertex


class Injector(
        AbstractPartitionManagedVertex,
        ReverseIPTagMulticastSourceMachineVertex):

    def __init__(self, n_keys, label):
        super(Injector, self).__init__(
            n_keys=n_keys, label=label, enable_injection=True)
