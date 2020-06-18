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
        self.partitions = {partition: Partition()
            for partition in globals.partitions_priority}

    def add_outgoing_partition(self, partition):
        self.partitions[partition].n_elements += 1

        # bubble the first key of each partition with a lower priority
        # upwards in the key space
        for partition_, priority in globals.partitions_priority.items():
            if priority < globals.partitions_priority[partition]:
                self.partitions[partition_].first_key += 1

    def generate_constraint(self, partition_identifier):
        partition = self.partitions[partition_identifier]
        key = partition.first_key + partition.next_key_offset
        partition.next_key_offset += 1

        return FixedKeyAndMaskConstraint([BaseKeyAndMask(
            key, globals.mask)])


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
