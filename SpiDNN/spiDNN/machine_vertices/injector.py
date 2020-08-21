from spinn_front_end_common.utility_models import \
    ReverseIPTagMulticastSourceMachineVertex

from .abstract_partition_managed_machine_vertex import \
    AbstractPartitionManagedMachineVertex


class Injector(
        AbstractPartitionManagedMachineVertex,
        ReverseIPTagMulticastSourceMachineVertex):
    """
    Wrapper around the RIPTMCS from the SpiNNaker toolchain.
    Makes sure the RIPTMCS gets the correct keys, by inheriting from
    AbstractPartitionManagedMachineVertex.
    """

    def __init__(self, layer, id, n_keys):
        self.layer = layer
        self.id = id

        label = "{}_{}".format(self.layer.label, self.id)

        ReverseIPTagMulticastSourceMachineVertex.__init__(
            self, label, n_keys=n_keys)
