import spinnaker_graph_front_end as front_end

from spinn_front_end_common.utilities.globals_variables import \
    get_simulator

from spinn_front_end_common.utilities.connections import \
    LiveEventConnection

from spinn_front_end_common.utility_models import \
    LivePacketGatherMachineVertex

from spinn_front_end_common.utilities.utility_objs import \
    LivePacketGatherParameters

from spinnman.messages.eieio import EIEIOType

from pacman.model.constraints.placer_constraints import \
    ChipAndCoreConstraint

from pacman.model.graphs.machine import MachineEdge

from spinn_utilities.socket_address import SocketAddress


from spiDNN.util import absolute_path_from_home, ReceivingLiveOutputProgress, \
    uint32t_to_float, float_to_uint32t

import spiDNN.globals as globals


import time


import numpy as np


class Model:
    def __init__(self):
        self.weights = []
        self._layers = []

    def add(self, layer, layer_name=None):
        # TODO: here control correct usage

        if layer_name is None:
            name = type(layer).__name__
            layer.name = "{}{}".format(len(self._layers), name)
        else:
            layer.name = layer_name

        # generate random weights
        if len(self._layers) > 0:
            source_layer = self._layers[-1]
            self.weights.append(layer.generate_weights(source_layer))

        self._layers.append(layer)
        return self

    def predict(self, X):
        # TODO: make sure X is np.float32

        # TODO: inject different end units depending whether
        #       simulation will train the model or do inference

        end_unit_args = LivePacketGatherParameters(
            port=globals.ack_port,
            hostname=globals.host,
            strip_sdp=True,
            message_type=EIEIOType.KEY_PAYLOAD_32_BIT,
            use_payload_prefix=False,
            payload_as_time_stamps=False
        )

        end_unit = LivePacketGatherMachineVertex(
            end_unit_args,
            "end_unit_lpg",
            constraints=[ChipAndCoreConstraint(x=0, y=0)]
        )

        # prediction only need lpg as out? softmax? handle softmax
        # in host or spawn an extra instance on board?
        #
        # do it only lpg. Better handled in python than putting board
        # under more pressure no?

        self._setup_front_end()

        self._add_db_sock()

        self._generate_machine_graph()

        # TODO: wrapper around end_unit so it can easily be integrated
        #      into _generate_machine_graph()
        front_end.add_machine_vertex_instance(end_unit)
        for source_neuron in self._layers[-1].neurons:
            front_end.add_machine_edge_instance(MachineEdge(
                source_neuron, end_unit, label="{}_to_{}".format(
                    source_neuron.label, end_unit.label
                )
            ), globals.partition_name)

        send_labels = self._layers[0].labels
        receive_labels = self._layers[-1].labels

        conn = LiveEventConnection(
            end_unit.label,
            receive_labels=receive_labels,
            send_labels=send_labels,
            machine_vertices=True
        )

        send_label_to_pos = \
            {label: i for i, label in enumerate(send_labels)}

        def injector_callback(label, conn):

            for i, x in enumerate(X):
                print("sending {} at step {} to neuron: {}".format(
                    x[send_label_to_pos[label]], i, label
                ))

                conn.send_event_with_payload(
                    label, 0, float_to_uint32t(x[send_label_to_pos[label]])
                )

                time.sleep(1)

            # conn.send_events_with_payloads(label,
            #    [(0, float(x[send_label_to_pos[label]])) for x in X])

        rlop = ReceivingLiveOutputProgress(X.shape[0], receive_labels)

        result = np.empty(
            (X.shape[0], len(receive_labels)), dtype=np.float32
        )

        # TODO
        def extractor_callback(label, _, val):
            val = uint32t_to_float(val)
            print("received val: {}, neuron: {}".format(val, label))
            x = rlop.received(label)
            y = rlop.label_to_pos(label)

            result[x, y] = val

            if rlop.simulation_finished:
                front_end.stop_run()

        for label in receive_labels:
            conn.add_receive_callback(label, extractor_callback)

        for label in send_labels:
            conn.add_start_resume_callback(label, injector_callback)

        front_end.run()

        front_end.stop()
        conn.close()

        return result

    def _setup_front_end(self):
        # + 1, because end_unit must be accounted for
        # TODO: incorporate maybe bigger end_units
        n_cores = self._all_atoms() + 1

        front_end.setup(
            n_chips_required=n_cores // globals.cores_per_chip,
            model_binary_folder=absolute_path_from_home(),
            machine_time_step=globals.machine_time_step,
            time_scale_factor=globals.time_scale_factor,
        )

        available_cores = \
            front_end.get_number_of_available_cores_on_machine()

        if available_cores <= n_cores:
            raise KeyError(
                "SpiNNaker doesn't have enough cores to run Model"
            )

    def _generate_machine_graph(self):
        self._init_neurons()
        self._connect_layers()

    def _init_neurons(self):
        # Input unit needs to know how many neurons it is connected
        # to
        #
        # TODO: how will this look with Conv2D????
        #
        self._layers[0].init_neurons(self._layers[1].atoms)

        for layer, weights in zip(self._layers[1:], self.weights):
            layer.init_neurons(weights)

    def _connect_layers(self):
        for i, layer in enumerate(self._layers[1:]):
            source_layer = self._layers[i]
            layer.connect(source_layer)

    def _all_atoms(self):
        return sum([layer.atoms for layer in self._layers])

    def _add_db_sock(self):
        database_socket = SocketAddress(
            listen_port=globals.ack_port,
            notify_host_name=globals.host,
            notify_port_no=globals.notify_port
        )

        get_simulator().add_socket_address(database_socket)
