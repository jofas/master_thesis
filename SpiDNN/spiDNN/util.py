from pacman.model.routing_info import BaseKeyAndMask


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


def __generate_keys_and_masks():
    mask = 0xffffffff
    for base_key in range(0x00000000, 0xffffffff):
        yield BaseKeyAndMask(base_key, mask)


__generator = __generate_keys_and_masks()
generate_keys_and_masks = lambda: next(__generator)


def uint32t_to_float(uint):
    bts = struct.pack("!I", uint)
    return struct.unpack("!f", bts)[0]


def float_to_uint32t(flt):
    bts = struct.pack("f", flt)
    return struct.unpack("I", bts)[0]


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
