from spinn_utilities.overrides import overrides

from pacman.model.graphs.machine import MachineVertex

from pacman.model.resources import ResourceContainer, VariableSDRAM

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


from .abstract_partition_managed_vertex import AbstractPartitionManagedVertex


import sys

import math

from enum import Enum

import struct


import numpy as np


class LossMachineVertex(
        AbstractPartitionManagedVertex,
        SimulatorVertex,
        MachineDataSpecableVertex):

    PARAMS_DATA_SIZE = 5 * BYTES_PER_WORD

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[("SYSTEM", 0), ("PARAMS", 1), ("KEYS", 2)])

    def __init__(self, layer, partition_manager):
        AbstractPartitionManagedVertex.__init__(
            self, partition_manager)

        SimulatorVertex.__init__(
            self, layer.label, "loss_machine_vertex.aplx")

        self.loss_function_id = globals.losses[layer.loss_fn]
        self.K = layer.K

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # Generate the system data region for simulation requirements
        generate_system_data_region(
            spec, self.DATA_REGIONS.SYSTEM.value, self,
            machine_time_step, time_scale_factor
        )

        spec.reserve_memory_region(
            region=self.DATA_REGIONS.PARAMS.value,
            size=self.PARAMS_DATA_SIZE,
            label="params"
        )

        spec.reserve_memory_region(
            region=self.DATA_REGIONS.KEYS.value,
            size=self.K * BYTES_PER_WORD,
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
            keys.append(routing_info.get_first_key_from_pre_vertex(
                self, partition.identifier))

        spec.switch_write_focus(
            region=self.DATA_REGIONS.PARAMS.value)
        spec.write_value(self.loss_function_id)
        spec.write_value(self.K)
        spec.write_value(min_pre_key)
        spec.write_value(min_y_key)
        spec.write_value(generate_offset(placement.p))

        spec.switch_write_focus(
            region=self.DATA_REGIONS.KEYS.value)
        spec.write_array(keys)

        spec.end_specification()

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT
                       + self.PARAMS_DATA_SIZE
                       + self.K * BYTES_PER_WORD)

        per_timestep_sdram = 0

        return ResourceContainer(
            sdram=VariableSDRAM(fixed_sdram, per_timestep_sdram))

    def __repr__(self):
        return self.label
