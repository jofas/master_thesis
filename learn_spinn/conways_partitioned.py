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

from pacman.model.graphs.machine import MachineEdge
import spinnaker_graph_front_end as front_end

from spinn_front_end_common.utilities.connections import \
    LiveEventConnection

from spinn_front_end_common.utility_models import \
    LivePacketGatherMachineVertex

from conways_basic_cell import ConwayBasicCell

import numpy as np

import csv

def to_askii(bool):
    return "X" if bool else "O"

arr_to_askii = np.vectorize(to_askii, otypes=[np.str])


active_states = [(2, 2), (3, 2), (3, 3), (4, 3), (2, 4)]

runtime = 50
# machine_time_step = 100

X_SIZE = 7
Y_SIZE = 7

n_chips = (X_SIZE * Y_SIZE) // 15


def main():
    # set up the front end and ask for the detected machines dimensions
    front_end.setup(
        n_chips_required=n_chips,
        model_binary_folder=os.path.dirname(__file__)
    )

    # figure out if machine can handle simulation
    cores = (front_end
        .get_number_of_available_cores_on_machine())

    if cores <= (X_SIZE * Y_SIZE):
        raise KeyError("Don't have enough cores to run simulation")

    # contain the vertices for the connection aspect
    vertices = [[None for _ in range(X_SIZE)] for _ in range(Y_SIZE)]
    streamers = np.array(
        [[None for _ in range(X_SIZE)] for _ in range(Y_SIZE)]
    )

    # build vertices
    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):
            vert = ConwayBasicCell(
                "cell_{}".format((x * X_SIZE) + y),
                (x, y) in active_states)

            streamer = LivePacketGatherMachineVertex(
                "streamer_{}".format((x * X_SIZE) + y),
                port = 19999,
                hostname="192.168.2.200"
            )

            front_end.add_machine_vertex_instance(vert)
            front_end.add_machine_vertex_instance(streamer)

            vertices[x][y] = vert
            streamers[x, y] = streamer

    # build edges {{{
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
                front_end.add_machine_edge_instance(
                    MachineEdge(
                        vertices[x][y], vertices[dest_x][dest_y],
                        label=compass), ConwayBasicCell.PARTITION_ID)

            # connect streamer with vertices
            front_end.add_machine_edge_instance(
                MachineEdge(
                    vertices[x][y],
                    streamers[x, y],
                    label="edge_{}".format(vertices[x][y].label)
                ),
                ConwayBasicCell.PARTITION_ID
            )
    # }}}

    labels = [s.label for s in streamers.flatten()]
    conn = LiveEventConnection(
       None, receive_labels=labels#, local_port=19995
    )

    def cb(label, time, stuff):
        print("received: {}, {}, {}".format(label, time, stuff))

    for label in labels: conn.add_receive_callback(label, cb)

    # run the simulation
    front_end.run(runtime)

    conn.close()

    """
    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):
            data = streamers[x][y].get_provenance_data_from_machine(
                front_end.transceiver(),
                front_end.placements().get_placement_of_vertex(
                    streamers[x][y]
                )
            )

            print(x, y, data)
    """

    # get recorded data
    recorded_data = np.empty((X_SIZE, Y_SIZE, runtime), dtype=np.int32)

    # get the data per vertex
    for x in range(0, X_SIZE):
        for y in range(0, Y_SIZE):
            recorded_data[x, y, :] = vertices[x][y].get_data(
                front_end.buffer_manager(),
                front_end.placements().get_placement_of_vertex(vertices[x][y]))


    #export_data(recorded_data)
    check_correctness(recorded_data)
    #visualize_conways(recorded_data)

    # clear the machine
    front_end.stop()


def visualize_conways(data):
    data = arr_to_askii(data)

    for time in range(0, runtime):
        print("at time {}\n{}".format(
            time, "".join([ "".join(data[:,y,time]) + "\n"
                for y in range(X_SIZE - 1, 0, -1)])
        ))


def export_data(data):
    with open("test.csv", "w") as f:
        w = csv.writer(f)
        for time in range(0, runtime):
            w.writerow(data[:,:,time].flatten())


def import_data():
    with open("test.csv", "r") as f:
        r = csv.reader(f)
        return np.array([row for row in r], dtype=np.int32)


def check_correctness(data):
    generated_output = np.array([data[:,:,time].flatten()
        for time in range(0, runtime)])

    correct_output = import_data()

    assert (correct_output == generated_output).all()


if __name__ == "__main__":
    main()
