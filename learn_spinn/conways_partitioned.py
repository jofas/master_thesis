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
    LivePacketGatherMachineVertex, \
    ReverseIPTagMulticastSourceMachineVertex

from spinn_front_end_common.utilities.utility_objs import \
    LivePacketGatherParameters

from spinn_front_end_common.interface.interface_functions import \
    ApplicationFinisher

from spinn_utilities.socket_address import SocketAddress

from conways_basic_cell import ConwayBasicCell

import numpy as np

import csv

from threading import Thread, Condition


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

    stream_in  = add_reverse_ip_tag_vertex("stream_in_instance")
    stream_out = add_lpg_machine_vertex("stream_out_instance")

    build_edges(vertices, stream_in, stream_out)

    add_db_sock()

    labels = [cc.label for cc in vertices.flatten()]

    conn = LiveEventConnection(
       stream_out.label, receive_labels=labels,
       send_labels=["stream_in_instance"],
       machine_vertices = True
    )

    # this stuff is needed for processing the received states
    receive_counter = {l : 0 for l in labels}
    receive_counter["overall"] = 0
    receive_counter["finished"] = False

    map_label_to_pos = {}
    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):
            map_label_to_pos[vertices[x, y].label] = (x, y)

    recorded_states = np.empty((X_SIZE, Y_SIZE, runtime), dtype=np.int32)

    def receive_state_callback(label, _, state): # {{{
        z = receive_counter[label]

        # application finisher takes longer that the simulation for
        # another time step. That way too many packets are received
        # which we don't want to add in our recorded data.
        if z < 50:
            x, y = map_label_to_pos[label]

            recorded_states[x, y, z] = state

            receive_counter[label] += 1
            receive_counter["overall"] += 1

            #print("received: {} at timestep {}: {}".format(label, z + 1, state))

        elif receive_counter["overall"] == len(labels) * 50 \
                and not receive_counter["finished"]:

            receive_counter["finished"] = True

            print("FINISHING SIMULATION")

            ApplicationFinisher()(
                front_end._sim()._load_outputs["APPID"],
                front_end.transceiver(),
                front_end._sim()._load_outputs["ExecutableTypes"]
            )
            front_end.stop_run()
    # }}}

    def send_state_callback(label, conn):
        print("CONNNNNNNN ", conn._atom_id_to_key)

        #for label in labels:
        #    state = int(bool(vertices[map_label_to_pos[label]]._state))
        conn.send_events_with_payloads(label, [(0, 0) for _ in labels])

    for label in labels:
        conn.add_receive_callback(label, receive_state_callback)

    conn.add_start_resume_callback( "stream_in_instance"
                                  , send_state_callback )

    front_end.run(None)

    front_end.stop()

    check_correctness(recorded_states)
    #visualize_conways(recorded_states)

    conn.close()


def build_edges(cc_machine_vertices, stream_in, stream_out): # {{{
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
                stream_in,
                label="stream_in_edge_{}".format(
                    cc_machine_vertices[x, y].label
                )
            ), ConwayBasicCell.PARTITION_ID)

            front_end.add_machine_edge_instance(MachineEdge(
                cc_machine_vertices[x, y],
                stream_out,
                label="stream_out_edge_{}".format(
                    cc_machine_vertices[x, y].label
                )
            ), ConwayBasicCell.PARTITION_ID)
# }}}


def add_cc_machine_vertices(): # {{{
    vertices = np.array([[None for _ in range(Y_SIZE)] for _ in range(X_SIZE)])

    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):
            vert = ConwayBasicCell(
                "cell_{}".format((x * X_SIZE) + y),
                (x, y) in active_states
            )

            front_end.add_machine_vertex_instance(vert)
            vertices[x, y] = vert

    return vertices
# }}}


def add_reverse_ip_tag_vertex(label): # {{{
    stream_in = ReverseIPTagMulticastSourceMachineVertex(
        n_keys = X_SIZE * Y_SIZE,
        label  = label,
        constraints=[ChipAndCoreConstraint(x=0, y=0)],
        enable_injection = True
    )

    front_end.add_machine_vertex_instance(stream_in)
    return stream_in
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

    stream_out = LivePacketGatherMachineVertex(
        args, label, constraints=[ChipAndCoreConstraint(x=0, y=0)],
    )

    front_end.add_machine_vertex_instance(stream_out)
    return stream_out
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
