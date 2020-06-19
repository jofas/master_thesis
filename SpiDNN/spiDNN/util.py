from pacman.model.routing_info import BaseKeyAndMask

from pacman.model.constraints.key_allocator_constraints import \
    FixedKeyAndMaskConstraint

from pacman.model.graphs.machine import MachineEdge

import spinnaker_graph_front_end as front_end


import spiDNN.globals as globals


import os

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


def generate_machine_edge(source, dest, partition):
    return MachineEdge(source, dest, label="{}_{}_to_{}".format(
        partition, source.label, dest.label))


def add_machine_edge_instance(source, dest, partition):
    front_end.add_machine_edge_instance(generate_machine_edge(
        source, dest, partition), partition)


def uint32t_to_float(uint):
    bts = struct.pack("!I", uint)
    return struct.unpack("!f", bts)[0]


def float_to_uint32t(flt):
    bts = struct.pack("f", flt)
    return struct.unpack("I", bts)[0]


class Partition:
    def __init__(self):
        self.n_elements = 0
        self.first_key = 0
        self.next_key_offset = 0


class PartitionManager:
    def __init__(self):
        self.partitions = {
            partition: Partition()
            for partition in globals.partitions_priority}
        self.priorities = {
            key: value
            for key, value in globals.partitions_priority.items()}
        self.priorities_reverse = {
            value: key
            for key, value in globals.partitions_priority.items()}
        self.next_priority = -1

    def add_outgoing_partition(self, partition_identifier):
        if partition_identifier not in self.partitions:
            self._add_partition(partition_identifier)

        self.partitions[partition_identifier].n_elements += 1

        # bubble the first key of each partition with a lower priority
        # upwards in the key space
        partition_priority = self.priorities[partition_identifier]
        for partition_, priority in self.priorities.items():
            if priority < partition_priority:
                self.partitions[partition_].first_key += 1

    def generate_constraint(self, partition_identifier):
        partition = self.partitions[partition_identifier]
        key = partition.first_key + partition.next_key_offset
        partition.next_key_offset += 1

        return FixedKeyAndMaskConstraint([BaseKeyAndMask(
            key, globals.mask)])

    def _add_partition(self, partition_identifier):
        lowest_priority_partition = self.partitions[
            self.priorities_reverse[self.next_priority + 1]]

        first_key = lowest_priority_partition.first_key \
            + lowest_priority_partition.n_elements

        new_partition = Partition()
        new_partition.first_key = first_key

        self.partitions[partition_identifier] = new_partition

        self.priorities[partition_identifier] = self.next_priority
        self.priorities_reverse[self.next_priority] = partition_identifier

        self.next_priority -= 1


class ReceivingLiveOutputProgress:
    def __init__(self, receive_n_times, receive_labels):
        self._receive_counter = {label: 0 for label in receive_labels}

        self._received_overall = 0

        self._receive_n_times = len(receive_labels) * receive_n_times

        self._label_to_pos = \
            {label: i for i, label in enumerate(receive_labels)}

        self._lock_overall = Lock()

    def received(self, label):
        current = self._receive_counter[label]
        self._receive_counter[label] += 1
        self._received_overall += 1
        return current

    def label_to_pos(self, label):
        return self._label_to_pos[label]

    @property
    def simulation_finished(self):
        with self._lock_overall:
            return self._received_overall == self._receive_n_times
