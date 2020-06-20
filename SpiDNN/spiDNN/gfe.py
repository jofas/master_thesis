import spinnaker_graph_front_end as front_end
from spinn_utilities.socket_address import SocketAddress
from spinn_front_end_common.utilities.globals_variables import \
    get_simulator

import spiDNN.globals as globals
import spiDNN.util as util


partition_manager = None


def setup(n_cores):
    global partition_manager
    partition_manager = util.PartitionManager()

    front_end.setup(
        n_chips_required=n_cores // globals.cores_per_chip,
        model_binary_folder=util.absolute_path_from_home(),
        machine_time_step=globals.machine_time_step,
        time_scale_factor=globals.time_scale_factor,
    )

    __add_db_sock()

    available_cores = \
        front_end.get_number_of_available_cores_on_machine()

    if available_cores <= n_cores:
        raise KeyError(
            "SpiNNaker doesn't have enough cores to run Model")


def run(time=None):
    front_end.run(time)


def stop():
    global partition_manager
    partition_manager = None

    front_end.stop()


def stop_run():
    front_end.stop_run()


def add_machine_vertex_instance(machine_vertex):
    front_end.add_machine_vertex_instance(machine_vertex)


def transceiver():
    return front_end.transceiver()


def placements():
    return front_end.placements()


def __add_db_sock():
    database_socket = SocketAddress(
        listen_port=globals.ack_port,
        notify_host_name=globals.host,
        notify_port_no=globals.notify_port
    )

    get_simulator().add_socket_address(database_socket)
