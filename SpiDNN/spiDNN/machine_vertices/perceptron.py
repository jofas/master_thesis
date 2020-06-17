from spinn_utilities.overrides import overrides

from pacman.model.graphs.machine import MachineVertex

from pacman.model.resources import ResourceContainer, VariableSDRAM

from pacman.model.constraints.key_allocator_constraints import \
    FixedKeyAndMaskConstraint

from pacman.utilities.utility_calls import is_single

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


from spiDNN.util import generate_offset, generate_keys_and_masks

import spiDNN.globals as globals


import sys

import math

from enum import Enum

import struct


import numpy as np


class AbstractPerceptronBase(SimulatorVertex, MachineDataSpecableVertex):

    BASE_PARAMS_DATA_SIZE = 5 * BYTES_PER_WORD

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[("SYSTEM", 0), ("BASE_PARAMS", 1), ("WEIGHTS", 2),
               ("INSTANCE_PARAMS", 3)])

    def __init__(self, layer, id, weights, executable,
                 instance_param_data_size):
        super(AbstractPerceptronBase, self).__init__(
            "{}_{}".format(layer.name, id), executable)

        self.weights = weights
        self._weight_container_size = len(self.weights) * BYTES_PER_WORD
        self._instance_param_data_size = instance_param_data_size

    def extract_weights(self, transceiver, placement):
        weights_region_base_address = locate_memory_region_for_placement(
            placement, self.DATA_REGIONS.WEIGHTS.value, transceiver)

        raw_data = transceiver.read_memory(
            placement.x, placement.y,
            weights_region_base_address,
            self._weight_container_size)

        unpacked_data = struct.unpack("<{}f".format(
            self.weights.shape[0]), raw_data)

        self.weights = np.array(unpacked_data, dtype=np.float32)

        return self.weights

    def abstract_generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        self._generate_data_regions(spec, machine_time_step,
                                    time_scale_factor)

        self._write_base_params(spec, machine_graph, routing_info, placement)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.WEIGHTS.value)
        spec.write_array(self.weights, data_type=DataType.FLOAT_32)

    def _generate_data_regions(self, spec, machine_time_step,
                               time_scale_factor):

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

    def _write_base_params(self, spec, machine_graph, routing_info, placement):
        edges = list(machine_graph.get_edges_ending_at_vertex(self))

        # smallest key from previous layer
        min_pre_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.partition_name) for edge in edges])

        key = routing_info.get_first_key_from_pre_vertex(
            self, globals.partition_name)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.BASE_PARAMS.value)
        spec.write_value(0 if key is None else 1)
        spec.write_value(0 if key is None else key)
        spec.write_value(min_pre_key)
        spec.write_value(generate_offset(placement.p))
        spec.write_value(len(self.weights))

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT
                       + self.BASE_PARAMS_DATA_SIZE
                       + self._instance_param_data_size
                       + self._weight_container_size)

        per_timestep_sdram = 0

        return ResourceContainer(
            sdram=VariableSDRAM(fixed_sdram, per_timestep_sdram))

    def __repr__(self):
        return self.label


class Perceptron(AbstractPerceptronBase):

    INSTANCE_PARAMS_DATA_SIZE = 1 * BYTES_PER_WORD
    EXECUTABLE = "perceptron.aplx"

    def __init__(self, layer, id, weights):
        super(Perceptron, self).__init__(
            layer, id, weights, self.EXECUTABLE,
            self.INSTANCE_PARAMS_DATA_SIZE)

        self._activation_function_id = globals.activations[layer.activation]

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # check got right number of keys and edges going into me
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)

        if not is_single(partitions):
            raise ConfigurationException(
                "Can only handle one type of partition.")

        super(Perceptron, self).abstract_generate_machine_data_specification(
            spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.INSTANCE_PARAMS.value)
        spec.write_value(self._activation_function_id)

        spec.end_specification()


class SoftmaxPerceptron(AbstractPerceptronBase,
                        AbstractProvidesOutgoingPartitionConstraints):

    INSTANCE_PARAMS_DATA_SIZE = 3 * BYTES_PER_WORD
    EXECUTABLE = "softmax_perceptron.aplx"

    def __init__(self, layer, id, weights, softmax_partition_identifier):
        super(SoftmaxPerceptron, self).__init__(
            layer, id, weights, self.EXECUTABLE,
            self.INSTANCE_PARAMS_DATA_SIZE)

        self._softmax_partition_identifier = softmax_partition_identifier
        self._layer = layer

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # check got right number of keys and edges going into me
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)

        if not len(partitions) == 2:
            raise ConfigurationException(
                "Can only handle global and softmax partition.")

        super(SoftmaxPerceptron, self)\
            .abstract_generate_machine_data_specification(
                spec, placement, machine_graph, routing_info, iptags,
                reverse_iptags, machine_time_step, time_scale_factor)

        softmax_partition = \
            list(filter(lambda x: x != globals.partition_name, partitions))[0]

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, softmax_partition.identifier))

        min_softmax_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, softmax_partition.identifier) for edge in edges])

        softmax_key = routing_info.get_first_key_from_pre_vertex(
            self, softmax_partition.identifier)

        spec.switch_write_focus(
            region=self.DATA_REGIONS.INSTANCE_PARAMS.value)
        spec.write_value(softmax_key)
        spec.write_value(min_softmax_key)
        spec.write_value(self._layer.atoms)

        spec.end_specification()

    @overrides(AbstractProvidesOutgoingPartitionConstraints
               .get_outgoing_partition_constraints)
    def get_outgoing_partition_constraints(self, partition):
        if partition.identifier == self._softmax_partition_identifier:
            return [FixedKeyAndMaskConstraint([generate_keys_and_masks()])]
        return []
