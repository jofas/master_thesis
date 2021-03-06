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
    """
    Neuron of a 1D convolutional layer.
    (!) BACKWARD PASS ON THIS BRANCH NOT WORKING (!)
    """

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
                (9 + self.trainable_params.n_elements) * BYTES_PER_WORD
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
            min_next_key = min([
                routing_info.get_first_key_from_pre_vertex(
                    edge.pre_vertex, globals.backward_partition)
                for edge in edges])

            next_layer = edges[0].pre_vertex.layer
            kernel_container = edges[0].pre_vertex

            n_errors = len(edges) * next_layer.n_filters
            n_next_layer_weights = n_errors * self.layer.n_filters

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
                # kernel is shared, so each neuron of next layer has
                # the same kernel/weights
                weights = edges[0].pre_vertex.weights
                next_layer_kernel_size = int(
                    len(weights) / next_layer.n_filters)

                # TODO: get backward pass to work
                #
                # all next_layer_weights look different (well not all
                # but there are a few possibilites (based on position
                # and offset)
                # I could load position onto board, which is not very
                # helpful or is it?
                #
                # + bool next_layer_has_kernel
                #
                # strides are already not working
                # because index key - min_next_key will be wrong in
                # the first place
                #
                # maybe use the key somehow instead of position
                #
                # Restart:
                #   problem: I need to share the next_layer_weights,
                #            when next_layer has filters
                #
                #            In itself not difficult. Ugly but not
                #            too difficult.
                #
                #            Solutions:
                #               * ugly sharing which will be horrible
                #                 but I can do that
                #
                #               * rework your backprop by sending
                #                 multiple but unused packets
                #
                #                 a perceptron would send N times
                #                 first one is for first neuron in
                #                 prev layer and so forth.
                #                 that means neurons would need
                #                 counter for each next_layer_neuron
                #                 and if counter ==
                #                   self.id - min_layer_key then
                #                 I use that and continue with back-
                #                 prop like before
                #
                #                 much, much nicer. Far less memory
                #                 pressure (currently
                #                 n_next_layer_weights * 2 (gradients))
                #                 and no shared state
                #
                #                 Much less weird. I'll try that in
                #                 a new branch before doing anything
                #                 else.
                #
                #                 but how about CNNs? min_layer_key
                #                 must be different no? They aren't
                #
                #
                # position = 1
                #
                # idx = <- receive key = 0x0 => + position = 1
                #
                # idx * self.n_filters:+1 => [.9, 1.]
                #
                # idx = <- receive key = 1x0 => + position = 2
                #          % self.n_filters (== next_layer.n_channels)
                #          = 0
                #
                # idx * self.n_filters:+1 => [.7, .8]
                #
                # idx = <- receive key = 0x1 => + position = 1
                #          + 1 * next_layer_kernel_size
                #
                # idx * self.n_filters:+1 => [.9, 1.]
                #
                # idx = <- receive key = 1x0 => + position = 2
                #          % self.n_filters = 0
                #
                # idx * self.n_filters:+1 => [.7, .8]
                #
                # currently:
                # [.9, 1., 1.5, 1.6, .7, .8, 1.3, 1.4]
                # [.7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
                #  ------
                #  self.n_filters (next_layer.n_channels)
                #  ----------
                #  kernel_size
                #  ------------------------
                #  kernel

                i = 0
                for edge in edges:
                    # TODO: this could be your achilles heel again
                    position = self.id % next_layer.kernel_shape[0] \
                        + edge.pre_vertex.lower_padding

                    filter = 0
                    for _ in range(0, next_layer.n_filters):
                        idx = position * next_layer.n_channels + filter

                        next_layer_weights[i:i+self.layer.n_filters] = \
                            weights[idx:idx+self.layer.n_filters]

                        filter += next_layer_kernel_size
                        i += self.layer.n_filters

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
        spec.write_value(len(edges))

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
