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

    PARAMS_DATA_SIZE = 7 * BYTES_PER_WORD

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[("SYSTEM", 0), ("PARAMS", 1), ("WEIGHTS", 2)]
    )

    def __init__(self, layer, id, weights):

        super(Perceptron, self).__init__(
            "{}_{}".format(layer.name, id),
            "perceptron.aplx"
        )

        self.weights = np.array(weights, dtype=np.float32)
        self._weight_container_size = len(self.weights) * BYTES_PER_WORD
        self._activation_function_id = globals.activations[layer.activation]
        self._pre_layer_activation_function_id = None

    @inject_items({"data_n_time_steps": "DataNTimeSteps"})
    @overrides(
        MachineDataSpecableVertex.generate_machine_data_specification,
        additional_arguments={"data_n_time_steps"})
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor,
            data_n_time_steps):

        # Generate the system data region for simulation requirements
        generate_system_data_region(
            spec, self.DATA_REGIONS.SYSTEM.value, self,
            machine_time_step, time_scale_factor
        )

        # reserve memory regions
        spec.reserve_memory_region(
            region=self.DATA_REGIONS.PARAMS.value,
            size=self.PARAMS_DATA_SIZE,
            label="params"
        )

        spec.reserve_memory_region(
            region=self.DATA_REGIONS.WEIGHTS.value,
            size=self._weight_container_size,
            label="weights"
        )

        # check got right number of keys and edges going into me
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)

        if not is_single(partitions):
            raise ConfigurationException(
                "Can only handle one type of partition.")

        edges = list(machine_graph.get_edges_ending_at_vertex(self))

        min_pre_key = sys.maxsize

        for edge in edges:
            pre_key = routing_info.get_first_key_from_pre_vertex(
                edge.pre_vertex, globals.partition_name)

            if pre_key < min_pre_key:
                min_pre_key = pre_key

            # TODO: recurrent neurons???
            if edge.pre_vertex == self:
                raise ConfigurationException(
                    "I'm connected to myself, this is deemed an error"
                    " please fix.")

        key = routing_info.get_first_key_from_pre_vertex(
            self, globals.partition_name)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.PARAMS.value)

        # has_key
        spec.write_value(0 if key is None else 1)

        spec.write_value(0 if key is None else key)

        spec.write_value(min_pre_key)

        offset = generate_offset(placement.p)
        spec.write_value(offset)

        spec.write_value(len(self.weights))

        spec.write_value(self._activation_function_id)

        spec.write_value(self._pre_layer_activation_function_id)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.WEIGHTS.value)

        spec.write_array(self.weights, data_type=DataType.FLOAT_32)

        spec.end_specification()

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT + self.PARAMS_DATA_SIZE
                                                + self._weight_container_size)
        per_timestep_sdram = 0
        return ResourceContainer(
            sdram=VariableSDRAM(fixed_sdram, per_timestep_sdram))

    def set_pre_layer_activation(self, pre_layer):
        self._pre_layer_activation_function_id = \
            globals.activations[pre_layer.activation]

    def __repr__(self):
        return self.label
