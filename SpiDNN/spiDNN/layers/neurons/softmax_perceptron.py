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


class SoftmaxPerceptron(SimulatorVertex, MachineDataSpecableVertex):

    PARAMS_DATA_SIZE = 7 * BYTES_PER_WORD

    DATA_REGIONS = Enum(
        value="DATA_REGIONS",
        names=[("SYSTEM", 0), ("PARAMS", 1), ("WEIGHTS", 2)]
    )

    def __init__(self, layer, id, weights):

        super(SoftmaxPerceptron, self).__init__(
            "{}_{}".format(layer.name, id),
            "softmax_perceptron.aplx"
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

        if not len(partitions) == 2:
            raise ConfigurationException(
                "Can only handle global and softmax partition.")

        softmax_partition = \
            list(filter(lambda x: x != globals.partition_name, partitions))[0]

        # routing info should give me a base key for my partition ->
        # on board
        #
        # also own key from softmax partition

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.partition_name))

        # smallest key from previous layer
        min_pre_key = min([
            routing_info.get_first_key_from_pre_vertex(
                edge.pre_vertex, globals.partition_name
            ) for edge in edges
        ])

        min_softmax_key = \
            routing_info.get_first_key_from_partition(softmax_partition)

        print(dir(routing_info))
        for partition in partitions:
            print(partition.identifier)
            print(routing_info.get_first_key_from_partition(partition))
        #print(help(routing_info.get_first_key_for_edge))

        #raise Exception("meh. Me debugging")

        # TODO: continue here making softmax partition work just
        #       striding that shit

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, softmax_partition.identifier))

        print(dir(routing_info))
        for i, edge in enumerate(edges):
            print(i, " ", routing_info.get_routing_info_for_edge(edge).get_keys())

        key = routing_info.get_first_key_from_pre_vertex(
            self, globals.partition_name)

        softmax_key = routing_info.get_first_key_from_pre_vertex(
            self, softmax_partition.identifier)

        print(min_pre_key, " ", min_softmax_key)
        print(key, " ", softmax_key)

        raise Exception("meh. Me debugging")

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

