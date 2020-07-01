from pacman.model.routing_info import BaseKeyAndMask
from pacman.model.constraints.key_allocator_constraints import \
    FixedKeyAndMaskConstraint
from data_specification.enums import DataType

import spiDNN.globals as globals

import os
import sys
from threading import Lock
import struct
import math


def absolute_path_from_home(relative_path=None):
    home_path = os.path.dirname(__file__).split("/")[:-1]

    if relative_path is None:
        return "/".join(home_path)

    if relative_path[0] == "/":
        relative_path = relative_path[1:]

    return "/".join(home_path + relative_path.split("/"))


def generate_offset(processor):
    return int(
        math.ceil(globals.max_offset / globals.cores_per_chip) * processor
    )


def uint32t_to_float(uint):
    bts = struct.pack("!I", uint)
    return struct.unpack("!f", bts)[0]


def float_to_uint32t(flt):
    bts = struct.pack("f", flt)
    return struct.unpack("I", bts)[0]


class TrainableParams:
    n_elements = 4

    def __init__(self, epochs, epoch_size, batch_size, learning_rate):
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def write_to_spec(self, spec):
        spec.write_value(self.epochs)
        spec.write_value(self.epoch_size)
        spec.write_value(self.batch_size)
        spec.write_value(self.learning_rate, data_type=DataType.FLOAT_32)


class Partition:
    def __init__(self, identifier):
        self.identifier = identifier
        self.first_key = 0

        self.machine_vertices = {}
        self.constraint_generated = []

    def add(self, machine_vertex):
        """
        Adds the machine_vertex to the list of alrady seen machine
        vertices.

        Returns True if machine_vertex has not been touched before,
        otherwise False is returned.
        """
        if machine_vertex.label not in self.machine_vertices:
            self.machine_vertices[machine_vertex.label] = \
                len(self.machine_vertices)
            self.constraint_generated.append(False)
            return True
        return False

    def get_key(self, machine_vertex):
        if machine_vertex.label not in self.machine_vertices:
            raise KeyError("""Partition {} has never seen MachineVertex
                {} as the source of an edge.""".format(
                self.identifier, machine_vertex.label))

        index = self.machine_vertices[machine_vertex.label]

        if self.constraint_generated[index]:
            raise KeyError(""""Partition {} has already generated the
                constraint for MachineVertex {}.""".format(
                self.identifier, machine_vertex.label))

        self.constraint_generated[index] = True
        return self.first_key + index

    def __len__(self):
        return len(self.machine_vertices)


class PartitionManager:
    def __init__(self):
        self.partitions = []
        self.partitions_lookup = {}

    def add_outgoing_partition(self, machine_vertex, partition_identifier):
        partition = self._get_partition(partition_identifier)

        if partition is None:
            partition = self._add_partition(partition_identifier)

        if partition.add(machine_vertex):
            # bubble the first key of each partition which was touched
            # after this partition upwards in the key space
            index = self.partitions_lookup[partition_identifier]
            if index < len(self.partitions) - 1:
                for partition in self.partitions[index + 1:]:
                    partition.first_key += 1

    def generate_constraint(self, machine_vertex, partition_identifier):
        partition = self._get_partition(partition_identifier)

        if partition is None:
            raise KeyError("I've never heard of parition: {}".format(
                partition_identifier))

        key = partition.get_key(machine_vertex)

        return FixedKeyAndMaskConstraint([BaseKeyAndMask(
            key, globals.mask)])

    def _get_partition(self, partition_identifier):
        if partition_identifier in self.partitions_lookup:
            return self.partitions[
                self.partitions_lookup[partition_identifier]]
        return None

    def _add_partition(self, partition_identifier):
        index = len(self.partitions)
        self.partitions_lookup[partition_identifier] = index

        new_partition = Partition(partition_identifier)

        if index > 0:
            new_partition.first_key = self.partitions[-1].first_key \
                + len(self.partitions[-1])

        self.partitions.append(new_partition)
        return new_partition


class LossExtractionManager:
    def __init__(self, epochs, epoch_size):
        self.epochs = epochs
        self.epoch_size = epoch_size

        self.epoch_counter = 1
        self.receive_counter = 0

    def receive(self, loss):
        loss = uint32t_to_float(loss)

        if self.receive_counter % self.epoch_size == 0:
            print("Epoch: {}/{}".format(self.epoch_counter, self.epochs))
            self.epoch_counter += 1

        self.receive_counter += 1

        if self.receive_counter % self.epoch_size == 0:
            print("{}/{} loss: {}".format(
                self.epoch_size, self.epoch_size, loss))
        else:
            sys.stdout.write("{}/{} loss: {}\r".format(
                self.receive_counter % self.epoch_size,
                self.epoch_size, loss))
            sys.stdout.flush()


class PingPongExtractionManager:
    def __init__(self, epochs, epoch_size, n_receive):
        self.epochs = epochs
        self.epoch_size = epoch_size
        self.n_receive = n_receive

        self.overall_receive_counter = 0
        self.receive_counter = 0
        self.lock = Lock()

    def receive(self):
        with self.lock:
            self.receive_counter += 1
            self.overall_receive_counter += 1
            return (self.overall_receive_counter - 1) // self.n_receive

    def reset(self):
        with self.lock:
            self.receive_counter = 0

    @property
    def received_all(self):
        with self.lock:
            return self.receive_counter == self.n_receive

    @property
    def simulation_finished(self):
        with self.lock:
            return self.overall_receive_counter == \
                self.epochs * self.epoch_size * self.n_receive
