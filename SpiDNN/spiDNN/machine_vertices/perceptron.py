from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from pacman.model.resources import ResourceContainer, VariableSDRAM
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
    INSTANCE_PARAMS = 3
    TRAINABLE_PARAMS = 4
    BACKWARD_KEYS = 5


class AbstractPerceptronBase(
        AbstractPartitionManagedMachineVertex,
        SimulatorVertex,
        MachineDataSpecableVertex):

    BASE_PARAMS_DATA_SIZE = 5 * BYTES_PER_WORD

    def __init__(self, layer, id, weights, trainable, batch_size, executable,
                 instance_params_data_size):

        self.instance_params_data_size = instance_params_data_size

        self.weights = weights
        self.weight_container_size = len(self.weights) * BYTES_PER_WORD

        self.trainable = trainable
        self.batch_size = batch_size

        if self.trainable:
            assert self.batch_size is not None

            self.trainable_params_data_size = 3 * BYTES_PER_WORD
            self.backward_keys_container_size = \
                self.weight_container_size - BYTES_PER_WORD

            executable = "trainable_{}".format(executable)
        else:
            self.trainable_params_data_size = 0
            self.backward_keys_container_size = 0

        super(AbstractPerceptronBase, self).__init__(
            "{}_{}".format(layer.label, id), executable)

    def extract_weights(self):
        transceiver = gfe.transceiver()
        placement = gfe.placements().get_placement_of_vertex(self)

        weights_region_base_address = locate_memory_region_for_placement(
            placement, PerceptronDataRegions.WEIGHTS.value, transceiver)

        raw_data = transceiver.read_memory(
            placement.x, placement.y,
            weights_region_base_address,
            self._weight_container_size)

        unpacked_data = struct.unpack("<{}f".format(
            self.weights.shape[0]), raw_data)

        self.weights = np.array(unpacked_data, dtype=np.float32)

        return self.weights

    @overrides(MachineDataSpecableVertex.generate_machine_data_specification)
    def generate_machine_data_specification(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        self._generate_data_regions(spec, machine_time_step, time_scale_factor)

        self._write_base_params(spec, placement, machine_graph, routing_info)

        self._write_weights(spec)

        # needs to be implemented by the inheriting class
        self._write_instance_params(
            spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor)

        if self.trainable:
            self._write_trainable_params(spec, machine_graph, routing_info)

        spec.end_specification()

    def _generate_data_regions(self, spec, machine_time_step,
                               time_scale_factor):

        # Generate the system data region for simulation requirements
        generate_system_data_region(
            spec, PerceptronDataRegions.SYSTEM.value, self,
            machine_time_step, time_scale_factor)

        # reserve memory regions
        spec.reserve_memory_region(
            region=PerceptronDataRegions.BASE_PARAMS.value,
            size=self.BASE_PARAMS_DATA_SIZE,
            label="base_params")

        spec.reserve_memory_region(
            region=PerceptronDataRegions.WEIGHTS.value,
            size=self.weight_container_size,
            label="weights")

        spec.reserve_memory_region(
            region=PerceptronDataRegions.INSTANCE_PARAMS.value,
            size=self.instance_params_data_size,
            label="instance_params")

        if self.trainable:
            spec.reserve_memory_region(
                region=PerceptronDataRegions.TRAINABLE_PARAMS.value,
                size=self.trainable_params_data_size,
                label="trainable_params")

            spec.reserve_memory_region(
                region=PerceptronDataRegions.BACKWARD_KEYS.value,
                size=self.backward_keys_container_size,
                label="backward_keys")

    def _write_base_params(self, spec, placement, machine_graph, routing_info):
        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.forward_partition))

        min_pre_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.forward_partition) for edge in edges])

        key = routing_info.get_first_key_from_pre_vertex(
            self, globals.forward_partition)

        spec.switch_write_focus(
            region=PerceptronDataRegions.BASE_PARAMS.value)
        spec.write_value(0 if key is None else 1)
        spec.write_value(0 if key is None else key)
        spec.write_value(min_pre_key)
        spec.write_value(generate_offset(placement.p))
        spec.write_value(len(self.weights))

    def _write_weights(self, spec):
        spec.switch_write_focus(
            region=PerceptronDataRegions.WEIGHTS.value)
        spec.write_array(self.weights, data_type=DataType.FLOAT_32)

    def _write_trainable_params(
            self, spec, machine_graph, routing_info):

        # then backward keys -> outgoing partitions
        #
        # I know how many neurons are in prev layer (base_params)
        # len(self.weights) - 1 = n_weights - 1 = n_potentials
        #
        # write_array into backward_keys dataregion with all dem keys

        print(dir(machine_graph))
        print(dir(routing_info))

        edges = list(machine_graph.get_edges_ending_at_vertex(self))

        partitions = [
            machine_graph.get_outgoing_partition_for_edge(edge).identifier
            for edge in edges]

        min_next_key = None
        n_errors = 0

        for edge in edges:
            partition = \
                machine_graph.get_outgoing_partition_for_edge(edge).identifier

            if partition.startswith(globals.backward_partition_base_name):
                key = routing_info.get_first_key_from_pre_vertex(
                    edge.pre_vertex, partition)

                if min_next_key is None or key < min_next_key:
                    min_next_key = key

                n_errors += 1

        backward_keys = []

        partitions = [
            partition.identifier for partition in
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)
        ]

        # eeeehhhhmmmm.... make sure backward_keys[i] matches neuron I find
        # when I retract the forward_pass... but should be default
        for partition in partitions:
            print(partition)
            if partition.startswith(
                    globals.backward_partition_base_name):
                backward_keys.append(
                    routing_info.get_first_key_from_pre_vertex(
                        self, partition))

        # special case: first hidden layer which is connected to
        # pong
        n_neurons_in_prev_layer = len(self.weights - 1) - 1
        if len(backward_keys) == 1 and n_neurons_in_prev_layer > 1:
            backward_keys *= n_neurons_in_prev_layer

        print(backward_keys)
        # TODO: get backward_keys
        raise Exception("Meh")


        spec.switch_write_focus(
            region=PerceptronDataRegions.TRAINABLE_PARAMS.value)
        spec.write_value(self.batch_size)
        spec.write_value(backward_key)
        spec.write_value(min_next_key)
        spec.write_value(n_errors)

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        fixed_sdram = (SYSTEM_BYTES_REQUIREMENT
                       + self.BASE_PARAMS_DATA_SIZE
                       + self.weight_container_size
                       + self.instance_params_data_size
                       + self.trainable_params_data_size
                       + self.backward_keys_container_size)

        per_timestep_sdram = 0

        return ResourceContainer(
            sdram=VariableSDRAM(fixed_sdram, per_timestep_sdram))

    def __repr__(self):
        return self.label


class Perceptron(AbstractPerceptronBase):
    INSTANCE_PARAMS_DATA_SIZE = 1 * BYTES_PER_WORD
    EXECUTABLE = "perceptron.aplx"

    def __init__(self, layer, id, weights, trainable, batch_size):
        super(Perceptron, self).__init__(
            layer, id, weights, trainable, batch_size, self.EXECUTABLE,
            self.INSTANCE_PARAMS_DATA_SIZE)

        self._activation_function_id = globals.activations[layer.activation]

    def _write_instance_params(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # check got right number of keys and edges going into me
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)

        if not self.trainable and not len(partitions) == 1:
            raise ConfigurationException(
                "Can only handle forward partition.")

        spec.switch_write_focus(
            region=PerceptronDataRegions.INSTANCE_PARAMS.value)
        spec.write_value(self._activation_function_id)


class SoftmaxPerceptron(AbstractPerceptronBase):
    INSTANCE_PARAMS_DATA_SIZE = 3 * BYTES_PER_WORD
    EXECUTABLE = "softmax_perceptron.aplx"

    def __init__(self, layer, id, weights, trainable, batch_size):
        super(SoftmaxPerceptron, self).__init__(
            layer, id, weights, trainable, batch_size, self.EXECUTABLE,
            self.INSTANCE_PARAMS_DATA_SIZE)

        self._layer = layer

    def _write_instance_params(
            self, spec, placement, machine_graph, routing_info, iptags,
            reverse_iptags, machine_time_step, time_scale_factor):

        # check got right number of keys and edges going into me
        partitions = \
            machine_graph.get_outgoing_edge_partitions_starting_at_vertex(self)

        if not self.trainable and not len(partitions) == 2:
            raise ConfigurationException(
                "Can only handle forward and softmax partition.")

        edges = list(
            machine_graph.get_edges_ending_at_vertex_with_partition_name(
                self, globals.softmax_partition))

        min_softmax_key = min([routing_info.get_first_key_from_pre_vertex(
            edge.pre_vertex, globals.softmax_partition) for edge in edges])

        softmax_key = routing_info.get_first_key_from_pre_vertex(
            self, globals.softmax_partition)

        spec.switch_write_focus(
            region=PerceptronDataRegions.INSTANCE_PARAMS.value)
        spec.write_value(softmax_key)
        spec.write_value(min_softmax_key)
        spec.write_value(self._layer.n_neurons)
