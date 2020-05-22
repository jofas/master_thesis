# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os

from spinnman.messages.eieio import EIEIOType

from pacman.model.graphs.machine import MachineEdge
from pacman.model.constraints.placer_constraints import \
    ChipAndCoreConstraint

import spinnaker_graph_front_end as front_end

from spinn_front_end_common.utilities.connections import \
    LiveEventConnection

# 19999
from spinn_front_end_common.utilities.constants import NOTIFY_PORT

from spinn_front_end_common.utilities.globals_variables import \
    get_simulator

from spinn_front_end_common.utility_models import \
    LivePacketGatherMachineVertex

from spinn_front_end_common.utilities.utility_objs import \
    LivePacketGatherParameters

from spinn_utilities.socket_address import SocketAddress

from conways_basic_cell import ConwayBasicCell

import numpy as np

import csv

active_states = [(2, 2), (3, 2), (3, 3), (4, 3), (2, 4)]

runtime = 50
machine_time_step = 1000

X_SIZE = 7
Y_SIZE = 7

n_chips = (X_SIZE * Y_SIZE) // 15

ACK_PORT = 22222
HOST = "127.0.0.1"

def main():
    front_end.setup(
        n_chips_required=n_chips,
        model_binary_folder=os.path.dirname(__file__),
        machine_time_step=machine_time_step,
        time_scale_factor=100
    )

    check_board_size()

    vertices = add_cc_machine_vertices()

    streamer = add_lpg_machine_vertex("streamer_instance")

    build_edges(vertices, streamer)

    add_db_sock()

    labels = [cc.label for cc in vertices.flatten()]

    conn = LiveEventConnection(
       streamer.label, receive_labels=labels, machine_vertices = True
    )

    # this stuff is needed for processing the received states
    receive_counter = {l : 0 for l in labels}

    map_label_to_pos = {}
    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):
            map_label_to_pos[vertices[x, y].label] = (x, y)

    recorded_states = np.empty((X_SIZE, Y_SIZE, runtime), dtype=np.int32)

    def receive_state_callback(label, _, state): # {{{
        z    = receive_counter[label]
        x, y = map_label_to_pos[label]

        recorded_states[x, y, z] = state

        receive_counter[label] += 1

        print("received: {} at timestep {}: {}".format(label, z + 1, state))
    # }}}

    for label in labels:
        conn.add_receive_callback(label, receive_state_callback)

    #front_end.run(runtime)
    front_end.run_until_complete(runtime)
    #front_end.run_until_complete()

    front_end.stop()
    conn.close()

    check_correctness(recorded_states)
    #visualize_conways(recorded_states)


def build_edges(cc_machine_vertices, lpg_machine_vertex): # {{{
    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):

            positions = [
                (x, (y + 1) % Y_SIZE, "N"),
                ((x + 1) % X_SIZE,
                    (y + 1) % Y_SIZE, "NE"),
                ((x + 1) % X_SIZE, y, "E"),
                ((x + 1) % X_SIZE,
                    (y - 1) % Y_SIZE, "SE"),
                (x, (y - 1) % Y_SIZE, "S"),
                ((x - 1) % X_SIZE,
                    (y - 1) % Y_SIZE, "SW"),
                ((x - 1) % X_SIZE, y, "W"),
                ((x - 1) % X_SIZE,
                    (y + 1) % Y_SIZE, "NW")]

            for (dest_x, dest_y, compass) in positions:
                front_end.add_machine_edge_instance(MachineEdge(
                    cc_machine_vertices[x, y],
                    cc_machine_vertices[dest_x, dest_y],
                    label=compass
                ), ConwayBasicCell.PARTITION_ID)

            front_end.add_machine_edge_instance(MachineEdge(
                cc_machine_vertices[x, y],
                lpg_machine_vertex,
                label="stream_edge_{}".format(cc_machine_vertices[x, y].label)
            ), ConwayBasicCell.PARTITION_ID)
# }}}


def add_cc_machine_vertices(): # {{{
    vertices = np.array([[None for _ in range(Y_SIZE)] for _ in range(X_SIZE)])

    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):
            """
            if x * X_SIZE + y < 16:
                vert = ConwayBasicCell(
                    "cell_{}".format((x * X_SIZE) + y),
                    (x * X_SIZE + y) % 2 == 0, #(x, y) in active_states
                    constraints=[ChipAndCoreConstraint(x=0, y=1)],
                )
            else:
                vert = ConwayBasicCell(
                    "cell_{}".format((x * X_SIZE) + y),
                    (x * X_SIZE + y) % 2 == 0, #(x, y) in active_states
                    constraints=[ChipAndCoreConstraint(x=1, y=0)],
                )

            """
            vert = ConwayBasicCell(
                "cell_{}".format((x * X_SIZE) + y),
                #(x * X_SIZE + y) % 2 == 0,
                (x, y) in active_states
                #constraints=[ChipAndCoreConstraint(x=1, y=0)],
            )

            front_end.add_machine_vertex_instance(vert)
            vertices[x, y] = vert

    return vertices
# }}}


def add_lpg_machine_vertex(label): # {{{
    args = LivePacketGatherParameters(
        port = ACK_PORT,
        hostname = HOST,
        strip_sdp = True,
        message_type = EIEIOType.KEY_PAYLOAD_32_BIT,
        use_payload_prefix = False,
        payload_as_time_stamps = False,
    )

    streamer = LivePacketGatherMachineVertex(
        args, label, constraints=[ChipAndCoreConstraint(x=0, y=0)],
    )

    front_end.add_machine_vertex_instance(streamer)
    return streamer
# }}}


def add_db_sock(): # {{{
    database_socket = SocketAddress(
        listen_port=ACK_PORT,
        notify_host_name=HOST,
        notify_port_no=NOTIFY_PORT
    )

    get_simulator().add_socket_address(database_socket)
# }}}


def visualize_conways(data): # {{{
    arr_to_askii = np.vectorize(lambda x: "X" if x else "O", otypes=[np.str])

    data = arr_to_askii(data)

    for time in range(0, runtime):
        print("at time {}\n{}".format(
            time + 1, "".join([ "".join(data[:,y,time]) + "\n"
                for y in range(X_SIZE - 1, 0, -1)])
        ))
# }}}


def check_correctness(data): # {{{
    generated_output = np.array([data[:,:,time].flatten()
        for time in range(0, runtime)])

    correct_output = import_data()

    # I think correct_output is one farther up than generated_output
    # so correct_output[:-1,:] should be equal to
    # generated_output[1:,:] (removed the initial neighbor states from
    # the conway's cell so the new simulation is one timestep behind
    # the original program)
    assert (correct_output[:-1,:] == generated_output[1:,:]).all()
# }}}


def export_data(data): # {{{
    with open("test.csv", "w") as f:
        w = csv.writer(f)
        for time in range(0, runtime):
            w.writerow(data[:,:,time].flatten())
# }}}


def import_data(): # {{{
    with open("test.csv", "r") as f:
        r = csv.reader(f)
        return np.array([row for row in r], dtype=np.int32)
# }}}


def check_board_size(): # {{{
    cores = front_end.get_number_of_available_cores_on_machine()

    if cores <= (X_SIZE * Y_SIZE):
        raise KeyError("Don't have enough cores to run simulation")
# }}}


if __name__ == "__main__":
    main()
