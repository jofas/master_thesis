from spinn_utilities.overrides import overrides

from pacman.executor.injection_decorator import inject_items

from pacman.model.graphs.machine import MachineVertex

from pacman.model.resources import ResourceContainer, VariableSDRAM

from pacman.utilities.utility_calls import is_single

from spinn_front_end_common.utilities.constants import (
    SYSTEM_BYTES_REQUIREMENT, BYTES_PER_WORD)

from spinn_front_end_common.utilities.exceptions import ConfigurationException

from spinn_front_end_common.utilities.helpful_functions import (
    locate_memory_region_for_placement)

from spinn_front_end_common.abstract_models.impl import (
    MachineDataSpecableVertex)

from spinn_front_end_common.interface.buffer_management.buffer_models import (
    AbstractReceiveBuffersToHost)

from spinnaker_graph_front_end.utilities import SimulatorVertex

from spinnaker_graph_front_end.utilities.data_utils import (
    generate_system_data_region)

from data_specification.enums import DataType


from spiDNN.util import generate_offset

import spiDNN.globals as globals


import sys

import math

from enum import Enum

import struct


import numpy as np


class Perceptron(SimulatorVertex, MachineDataSpecableVertex):

    BASE_PARAMS_DATA_SIZE = 5 * BYTES_PER_WORD
    INSTANCE_PARAMS_DATA_SIZE = 1 * BYTES_PER_WORD

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[("SYSTEM", 0), ("BASE_PARAMS", 1), ("WEIGHTS", 2),
            ("INSTANCE_PARAMS", 3)]
    )

    def __init__(self, layer, id, weights):

        super(Perceptron, self).__init__(
            "{}_{}".format(layer.name, id),
            "perceptron.aplx"
        )

        self.weights = np.array(weights, dtype=np.float32)
        self._weight_container_size = len(self.weights) * BYTES_PER_WORD
        self._activation_function_id = globals.activations[layer.activation]

    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # Generate the system data region for simulation requirements
        generate_system_data_region(
            spec, self.DATA_REGIONS.SYSTEM.value, self,
            machine_time_step, time_scale_factor
        )

        # reserve memory regions
        spec.reserve_memory_region(
            region=self.DATA_REGIONS.BASE_PARAMS.value,
            size=self.BASE_PARAMS_DATA_SIZE,
            label="base_params"
        )

        spec.reserve_memory_region(
            region=self.DATA_REGIONS.WEIGHTS.value,
            size=self._weight_container_size,
            label="weights"
        )

        spec.reserve_memory_region(
            region=self.DATA_REGIONS.INSTANCE_PARAMS.value,
            size=self.INSTANCE_PARAMS_DATA_SIZE,
            label="instance_params"
        )

        # check got right number of keys and edges going into me
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)

        if not is_single(partitions):
            raise ConfigurationException(
                "Can only handle one type of partition.")

        edges = list(machine_graph.get_edges_ending_at_vertex(self))

        # smallest key from previous layer
        min_pre_key = min([
            routing_info.get_first_key_from_pre_vertex(
                edge.pre_vertex, globals.partition_name
            ) for edge in edges
        ])

        key = routing_info.get_first_key_from_pre_vertex(
            self, globals.partition_name)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.BASE_PARAMS.value)
        spec.write_value(0 if key is None else 1)
        spec.write_value(0 if key is None else key)
        spec.write_value(min_pre_key)
        spec.write_value(generate_offset(placement.p))
        spec.write_value(len(self.weights))

        spec.switch_write_focus(
            region=self.DATA_REGIONS.WEIGHTS.value)
        spec.write_array(self.weights, data_type=DataType.FLOAT_32)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.INSTANCE_PARAMS.value)
        spec.write_value(self._activation_function_id)

        spec.end_specification()

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT
            + self.BASE_PARAMS_DATA_SIZE
            + self.INSTANCE_PARAMS_DATA_SIZE
            + self._weight_container_size)

        per_timestep_sdram = 0

        return ResourceContainer(
            sdram=VariableSDRAM(fixed_sdram, per_timestep_sdram))

    def __repr__(self):
        return self.label
