from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, ConstantSDRAM
from spinnaker_graph_front_end.utilities import SimulatorVertex


class Conv1DNeuron(SimulatorVertex):
    def __init__(self, layer, id, weights):
        self.layer = layer
        self.id = id
        self.weights = weights

        super(Conv1DNeuron, self).__init__(
            "{}_{}".format(layer.label, self.id), "perceptron.aplx")

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = 0

        return ResourceContainer(sdram=ConstantSDRAM(fixed_sdram))


class Conv1DMeta(SimulatorVertex):
    def __init__(self, layer, weights, trainable_params):
        self.layer = layer
        self.weights = weights
        self.trainable_params = trainable_params

        super(Conv1DMeta, self).__init__(
            "{}_meta".format(layer.label), "perceptron.aplx")

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = 0

        return ResourceContainer(sdram=ConstantSDRAM(fixed_sdram))
