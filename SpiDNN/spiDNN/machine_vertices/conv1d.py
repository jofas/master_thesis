from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, ConstantSDRAM
from spinn_front_end_common.utilities.constants import (
    SYSTEM_BYTES_REQUIREMENT, BYTES_PER_WORD)
from spinn_front_end_common.utilities.exceptions import ConfigurationException
from spinn_front_end_common.utilities.helpful_functions import (
    locate_memory_region_for_placement)
from spinn_front_end_common.abstract_models import \
    AbstractProvidesNKeysForPartition
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
from .data_regions import DataRegions

import sys
import math
import struct

import numpy as np


class Conv1DNeuron(
        AbstractPartitionManagedMachineVertex,
        SimulatorVertex,
        MachineDataSpecableVertex):

    BASE_PARAMS_DATA_SIZE = 8 * BYTES_PER_WORD

    def __init__(self, layer, id, weights, trainable_params):
        executable = "conv1d.aplx"

        self.layer = layer
        self.id = id

        self.weights = weights
        self.weight_container_size = len(self.weights) * BYTES_PER_WORD

        self.key_container_size = self.layer.n_filters * BYTES_PER_WORD

        self.trainable_params = trainable_params

        self.lower_padding = 0
        self.upper_padding = 0

        if self.trainable_params is not None:
            self.trainable_params_data_size = \
                (8 + self.trainable_params.n_elements) * BYTES_PER_WORD
            executable = "trainable_{}".format(executable)
        else:
            self.trainable_params_data_size = 0

        super(Conv1DNeuron, self).__init__(
            "{}_{}".format(layer.label, self.id), executable)

    def extract_weights(self):
        transceiver = gfe.transceiver()
        placement = gfe.placements().get_placement_of_vertex(self)

        weights_region_base_address = locate_memory_region_for_placement(
            placement, DataRegions.WEIGHTS.value, transceiver)

        raw_data = transceiver.read_memory(
            placement.x, placement.y,
            weights_region_base_address,
            self.weight_container_size)

        unpacked_data = struct.unpack("<{}f".format(
            self.weights.shape[0]), raw_data)

        self.weights = np.array(unpacked_data, dtype=np.float32)

        return self.weights

    @overrides(AbstractProvidesNKeysForPartition
               .get_n_keys_for_partition)
    def get_n_keys_for_partition(self, partition):
        if not isinstance(partition, str):
            partition = partition.identifier

        if partition == globals.forward_partition and self.layer.flatten:
            return self.layer.n_filters
        return 1

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT
                       + self.BASE_PARAMS_DATA_SIZE
                       + self.key_container_size
                       + self.weight_container_size
                       + self.trainable_params_data_size)

        return ResourceContainer(sdram=ConstantSDRAM(fixed_sdram))

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # Generate the system data region for simulation requirements
        generate_system_data_region(
            spec, DataRegions.SYSTEM.value, self,
            machine_time_step, time_scale_factor)

        self._generate_and_write_base_params(
            spec, placement, machine_graph, routing_info)

        self._generate_and_write_keys(spec, routing_info)

        self._generate_and_write_weights(spec)

        if self.trainable_params is not None:
            self._generate_and_write_trainable_regions(
                spec, machine_graph, routing_info)

        spec.end_specification()

    def _generate_and_write_base_params(
            self, spec, placement, machine_graph, routing_info):
        spec.reserve_memory_region(
            region=DataRegions.BASE_PARAMS.value,
            size=self.BASE_PARAMS_DATA_SIZE,
            label="base_params")

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.forward_partition))

        min_pre_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.forward_partition) for edge in edges])

        spec.switch_write_focus(
            region=DataRegions.BASE_PARAMS.value)
        spec.write_value(min_pre_key)
        spec.write_value(generate_offset(placement.p))
        spec.write_value(self.layer.kernel_shape[0])
        spec.write_value(self.layer.n_channels)
        spec.write_value(self.layer.n_filters)
        spec.write_value(self.lower_padding)
        spec.write_value(self.upper_padding)
        spec.write_value(globals.activations[self.layer.activation])

    def _generate_and_write_keys(self, spec, routing_info):
        spec.reserve_memory_region(
            region=DataRegions.KEYS.value,
            size=self.key_container_size,
            label="keys")

        keys = routing_info.get_routing_info_from_pre_vertex(
            self, globals.forward_partition).get_keys()

        if len(keys) == 1:
            keys = [keys[0] for _ in range(0, self.layer.n_filters)]

        assert len(keys) == self.layer.n_filters

        spec.switch_write_focus(
            region=DataRegions.KEYS.value)
        spec.write_array(keys)

    def _generate_and_write_weights(self, spec):
        spec.reserve_memory_region(
            region=DataRegions.WEIGHTS.value,
            size=self.weight_container_size,
            label="weights")

        spec.switch_write_focus(
            region=DataRegions.WEIGHTS.value)
        spec.write_array(self.weights, data_type=DataType.FLOAT_32)

    def _generate_and_write_trainable_regions(
            self, spec, machine_graph, routing_info):
        # I have no idea how package initialization works, but I can't
        # import it in the global space, because spiDNN does not yet
        # have its subpackages as attributes
        from spiDNN.layers import Dense

        spec.reserve_memory_region(
            region=DataRegions.TRAINABLE_PARAMS.value,
            size=self.trainable_params_data_size,
            label="trainable_params")

        backward_key = routing_info.get_first_key_from_pre_vertex(
            self, globals.backward_partition)

        kernel_update_key = routing_info.get_first_key_from_pre_vertex(
            self, globals.kernel_update_partition)

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.kernel_update_partition))

        min_layer_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.kernel_update_partition)
            for edge in edges])

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.backward_partition))

        is_output_layer = len(edges) == 0

        if is_output_layer:
            raise Exception("Unimplemented")
        else:

            # TODO: if next_layer is Dense or Conv1D
            #
            # Dense => receive 1 but multiply with self.n_filters many
            #          next_layer_weights.
            #
            #          also how are weights extracted?
            #          next_layer_weights[id:id+self.n_filters]
            #
            # Conv1D => receive next_layer.n_filters and have
            #           next_layer.n_filters many next_layer_weights
            #
            #           how are weights extracted?
            #           ... that's a quite interesting question
            #
            # n_errors != sizeof(next_layer_weights)
            #
            # I'm a bitch and will just multiply n_errors if
            # n_next_layer_weights > n_errors
            #
            # additional params: kernel_update_key
            #                    n_next_layer_weights
            #
            #
            # NEXT: implement backprop on spinnaker

            min_next_key = min([
                routing_info.get_first_key_from_pre_vertex(
                    edge.pre_vertex, globals.backward_partition)
                for edge in edges])

            next_layer = edges[0].pre_vertex.layer
            kernel_container = edges[0].pre_vertex

            n_errors = len(edges) * next_layer.n_filters

            if type(next_layer) == Dense:
                n_next_layer_weights = n_errors * self.layer.n_filters
            else:
                n_next_layer_weights = n_errors

            self.next_layer_weights_container_size = \
                n_next_layer_weights * BYTES_PER_WORD

            next_layer_weights = np.empty(
                (n_next_layer_weights,), dtype=np.float32)

            if type(next_layer) == Dense:
                idx = self.id * self.layer.n_filters

                for i, edge in enumerate(edges):
                    next_layer_weights[i:i+self.layer.n_filters] = \
                        edge.pre_vertex.weights[
                            idx:idx + self.layer.n_filters]
            else:
                weights = edges[0].pre_vertex.weights
                next_layer_kernel_size = int(
                    len(weights) / next_layer.n_filters)

                i = 0
                for edge in edges:
                    # TODO: this could be your achilles heel again
                    position = self.id % next_layer.kernel_shape[0] \
                               + edge.pre_vertex.lower_padding

                    j = 0
                    for filter in range(0, next_layer.n_filters):
                        next_layer_weights[i] = \
                            weights[position * next_layer.n_channels + j]
                        j += next_layer_kernel_size
                        i += 1

            spec.reserve_memory_region(
                region=DataRegions.NEXT_LAYER_WEIGHTS.value,
                size=self.next_layer_weights_container_size,
                label="next_layer_weights")

            spec.switch_write_focus(
                region=DataRegions.NEXT_LAYER_WEIGHTS.value)
            spec.write_array(next_layer_weights, data_type=DataType.FLOAT_32)

        spec.switch_write_focus(
            region=DataRegions.TRAINABLE_PARAMS.value)
        spec.write_value(backward_key)
        spec.write_value(min_next_key)
        spec.write_value(n_errors)
        spec.write_value(int(is_output_layer))
        spec.write_value(kernel_update_key)
        spec.write_value(min_layer_key)
        spec.write_value(self.layer.n_neurons)
        spec.write_value(n_next_layer_weights)

        self.trainable_params.write_to_spec(spec)

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
