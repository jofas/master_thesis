from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, ConstantSDRAM
from pacman.model.constraints.key_allocator_constraints import \
    FixedKeyAndMaskConstraint
from spinn_front_end_common.utilities.constants import (
    SYSTEM_BYTES_REQUIREMENT, BYTES_PER_WORD)
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinn_front_end_common.utilities.helpful_functions import (
    locate_memory_region_for_placement)
from spinn_front_end_common.abstract_models import \
    AbstractProvidesOutgoingPartitionConstraints
from spinn_front_end_common.abstract_models.impl import (
    MachineDataSpecableVertex)
from spinnaker_graph_front_end.utilities import SimulatorVertex

from spinnaker_graph_front_end.utilities.data_utils import (
    generate_system_data_region)
from data_specification.enums import DataType

from spiDNN.util import generate_offset
import spiDNN.globals as globals

from .abstract_partition_managed_machine_vertex import \
    AbstractPartitionManagedMachineVertex
from .data_regions import DataRegions

import sys
import math
import struct

import numpy as np


class LossMachineVertex(
        AbstractPartitionManagedMachineVertex,
        SimulatorVertex,
        MachineDataSpecableVertex):

    PARAMS_DATA_SIZE = 7 * BYTES_PER_WORD

    def __init__(self, layer, trainable_params):
        super(LossMachineVertex, self).__init__(
            layer.label, "loss_machine_vertex.aplx")

        self.layer = layer
        self.trainable_params = trainable_params

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # Generate the system data region for simulation requirements
        generate_system_data_region(
            spec, DataRegions.SYSTEM.value, self,
            machine_time_step, time_scale_factor
        )

        spec.reserve_memory_region(
            region=DataRegions.BASE_PARAMS.value,
            size=self.PARAMS_DATA_SIZE,
            label="params"
        )

        spec.reserve_memory_region(
            region=DataRegions.KEYS.value,
            size=self.layer.K * BYTES_PER_WORD,
            label="keys"
        )

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.forward_partition))

        # smallest key from previous layer
        min_pre_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.forward_partition) for edge in edges])

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.y_partition))

        min_y_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.y_partition) for edge in edges])

        # all the unique partitions to the output layer
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)

        keys = []
        for partition in partitions:
            if partition.identifier != globals.backward_partition:
                keys.append(routing_info.get_first_key_from_pre_vertex(
                    self, partition.identifier))

        extractor_key = routing_info.get_first_key_from_pre_vertex(
            self, globals.backward_partition)

        spec.switch_write_focus(
            region=DataRegions.BASE_PARAMS.value)
        spec.write_value(extractor_key)
        spec.write_value(globals.losses[self.layer.loss])
        spec.write_value(self.layer.K)
        spec.write_value(min_pre_key)
        spec.write_value(min_y_key)
        spec.write_value(generate_offset(placement.p))
        spec.write_value(self.trainable_params.epoch_size)

        spec.switch_write_focus(
            region=DataRegions.KEYS.value)
        spec.write_array(keys)

        spec.end_specification()

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT
                       + self.PARAMS_DATA_SIZE
                       + self.layer.K * BYTES_PER_WORD)

        return ResourceContainer(sdram=ConstantSDRAM(fixed_sdram))

    def __repr__(self):
        return self.label
