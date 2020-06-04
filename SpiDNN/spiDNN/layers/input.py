import spinnaker_graph_front_end as front_end

from spinn_front_end_common.utility_models import \
    ReverseIPTagMulticastSourceMachineVertex


class Input:
    def __init__(self, atoms):
        self.atoms = atoms
        self.name = "uninitialized"
        self.neurons = []

    def init_neurons(self, atoms_next_layer):
        for i in range(0, self.atoms):
            neuron = ReverseIPTagMulticastSourceMachineVertex(
                n_keys=atoms_next_layer,
                label="{}_{}".format(self.name, i),
                enable_injection=True,
            )
            self.neurons.append(neuron)
            front_end.add_machine_vertex_instance(neuron)

    @property
    def labels(self):
        return [neuron.label for neuron in self.neurons]
