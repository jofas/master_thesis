from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, ConstantSDRAM
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

import spiDNN.gfe as gfe
import spiDNN.globals as globals
from spiDNN.util import generate_offset

from .abstract_partition_managed_machine_vertex import \
    AbstractPartitionManagedMachineVertex

import sys
import math
from enum import Enum
import struct

import numpy as np

class PerceptronDataRegions(Enum):
    SYSTEM = 0
    BASE_PARAMS = 1
    WEIGHTS = 2
    SOFTMAX_PARAMS = 3
    TRAINABLE_PARAMS = 4
    NEXT_LAYER_WEIGHTS = 5


class Perceptron(
        AbstractPartitionManagedMachineVertex,
        SimulatorVertex,
        MachineDataSpecableVertex):

    BASE_PARAMS_DATA_SIZE = 5 * BYTES_PER_WORD

    def __init__(
            self, layer, id, weights, trainable, batch_size, learning_rate):
        executable = "perceptron.aplx"

        self.layer = layer
        self.id = id

        self.weights = weights
        self.weight_container_size = len(self.weights) * BYTES_PER_WORD

        self.trainable = trainable
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if self.layer.activation == "softmax":
            self.softmax_params_data_size = 2 * BYTES_PER_WORD
            executable = "softmax_{}".format(executable)
        else:
            self.softmax_params_data_size = 0

        if self.trainable:
            assert self.batch_size is not None
            assert self.learning_rate is not None

            self.trainable_params_data_size = 6 * BYTES_PER_WORD
            executable = "trainable_{}".format(executable)
        else:
            self.trainable_params_data_size = 0

        # set during data_spec generation, because I need the machine
        # graph to collect the weights
        self.next_layer_weights_container_size = 0

        super(Perceptron, self).__init__(
            "{}_{}".format(layer.label, self.id), executable)

    def extract_weights(self):
        transceiver = gfe.transceiver()
        placement = gfe.placements().get_placement_of_vertex(self)

        weights_region_base_address = locate_memory_region_for_placement(
            placement, PerceptronDataRegions.WEIGHTS.value, transceiver)

        raw_data = transceiver.read_memory(
            placement.x, placement.y,
            weights_region_base_address,
            self.weight_container_size)

        unpacked_data = struct.unpack("<{}f".format(
            self.weights.shape[0]), raw_data)

        self.weights = np.array(unpacked_data, dtype=np.float32)

        return self.weights

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT
                       + self.BASE_PARAMS_DATA_SIZE
                       + self.weight_container_size
                       + self.softmax_params_data_size
                       + self.trainable_params_data_size
                       + self.next_layer_weights_container_size)

        return ResourceContainer(sdram=ConstantSDRAM(fixed_sdram))

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # Generate the system data region for simulation requirements
        generate_system_data_region(
            spec, PerceptronDataRegions.SYSTEM.value, self,
            machine_time_step, time_scale_factor)

        self._generate_and_write_base_params(
            spec, placement, machine_graph, routing_info)

        self._generate_and_write_weights(spec)

        if self.layer.activation == "softmax":
            self._generate_and_write_softmax_params(spec, routing_info)

        if self.trainable:
            self._generate_and_write_trainable_regions(
                spec, machine_graph, routing_info)

        spec.end_specification()

    def _generate_and_write_base_params(
            self, spec, placement, machine_graph, routing_info):
        spec.reserve_memory_region(
            region=PerceptronDataRegions.BASE_PARAMS.value,
            size=self.BASE_PARAMS_DATA_SIZE,
            label="base_params")

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.forward_partition))

        min_pre_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.forward_partition) for edge in edges])

        key = routing_info.get_first_key_from_pre_vertex(
            self, globals.forward_partition)

        spec.switch_write_focus(
            region=PerceptronDataRegions.BASE_PARAMS.value)
        spec.write_value(key)
        spec.write_value(min_pre_key)
        spec.write_value(generate_offset(placement.p))
        spec.write_value(len(self.weights))
        spec.write_value(globals.activations[self.layer.activation])

    def _generate_and_write_weights(self, spec):
        spec.reserve_memory_region(
            region=PerceptronDataRegions.WEIGHTS.value,
            size=self.weight_container_size,
            label="weights")

        spec.switch_write_focus(
            region=PerceptronDataRegions.WEIGHTS.value)
        spec.write_array(self.weights, data_type=DataType.FLOAT_32)

    def _generate_and_write_softmax_params(self, spec, routing_info):
        spec.reserve_memory_region(
            region=PerceptronDataRegions.SOFTMAX_PARAMS.value,
            size=self.softmax_params_data_size,
            label="softmax_params")

        softmax_key = routing_info.get_first_key_from_pre_vertex(
            self, globals.softmax_partition)

        spec.switch_write_focus(
            region=PerceptronDataRegions.SOFTMAX_PARAMS.value)
        spec.write_value(softmax_key)
        spec.write_value(self.layer.n_neurons)

    def _generate_and_write_trainable_regions(
            self, spec, machine_graph, routing_info):
        spec.reserve_memory_region(
            region=PerceptronDataRegions.TRAINABLE_PARAMS.value,
            size=self.trainable_params_data_size,
            label="trainable_params")

        backward_key = routing_info.get_first_key_from_pre_vertex(
            self, globals.backward_partition)

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.backward_partition))

        is_output_layer = len(edges) == 0

        if is_output_layer:
            edges = self \
                .get_edges_ending_at_vertex_where_partition_name_starts_with(
                    machine_graph, globals.backward_partition)

            assert len(edges) == 1

            out_edge = edges[0]
            partition = machine_graph.get_outgoing_partition_for_edge(
                out_edge).identifier

            min_next_key = routing_info.get_first_key_from_pre_vertex(
                out_edge.pre_vertex, partition)

            n_errors = 1
        else:
            min_next_key = min([
                routing_info.get_first_key_from_pre_vertex(
                    edge.pre_vertex, globals.backward_partition)
                for edge in edges])

            n_errors = len(edges)

            self.next_layer_weights_container_size = \
                len(edges) * BYTES_PER_WORD

            next_layer_weights = [
                edge.pre_vertex.weights[self.id] for edge in edges]

            spec.reserve_memory_region(
                region=PerceptronDataRegions.NEXT_LAYER_WEIGHTS.value,
                size=self.next_layer_weights_container_size,
                label="next_layer_weights")

            spec.switch_write_focus(
                region=PerceptronDataRegions.NEXT_LAYER_WEIGHTS.value)
            spec.write_array(next_layer_weights, data_type=DataType.FLOAT_32)

        spec.switch_write_focus(
            region=PerceptronDataRegions.TRAINABLE_PARAMS.value)
        spec.write_value(self.batch_size)
        spec.write_value(backward_key)
        spec.write_value(min_next_key)
        spec.write_value(n_errors)
        spec.write_value(int(is_output_layer))
        spec.write_value(self.learning_rate, data_type=DataType.FLOAT_32)

    def get_edges_ending_at_vertex_where_partition_name_starts_with(
            self, machine_graph, starts_with_str):

        edges = machine_graph.get_edges_ending_at_vertex(self)

        result = []

        for edge in edges:
            partition = machine_graph.get_outgoing_partition_for_edge(edge) \
                .identifier

            if partition.startswith(starts_with_str):
                result.append(edge)

        return result

    def __repr__(self):
        return self.label
